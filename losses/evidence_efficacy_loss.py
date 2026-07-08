"""
Evidence efficacy constraint adapter for CCEL-Net.

Design boundary
---------------
This file is NOT a standalone loss and should not implement CE or a manual
lambda-weighted objective such as:

    CE + lambda_eff * EvidenceEfficacyLoss

It only converts efficacy metrics computed by ``ccel.metrics.efficacy_metrics``
into constraint violations:

    class_violation_c = mask_c * [rho_c - psi_c]_+
    map_violation     = [rho_map - psi_map]_+      # only if enabled

The final objective and dual-variable update belong to
``ccel.losses.primal_dual_loss``:

    L = CE(b + e, y) + sum_c mu_c * class_violation_c
        + mu_map * map_violation

All formula-level metric computation must live in
``ccel.metrics.efficacy_metrics``. This adapter deliberately does NOT redefine
soft_confusion_matrix, chance_corrected_efficacy, or map-wise efficacy.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from ccel.metrics.efficacy_metrics import chance_corrected_efficacy

Tensor = torch.Tensor


def _make_class_vector(
    value: Union[float, Iterable[float], Tensor],
    num_classes: int,
    *,
    name: str,
) -> Tensor:
    """Convert scalar/list/tensor to a float tensor with shape [C]."""
    out = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if out.numel() == 1:
        out = out.repeat(num_classes)
    if out.numel() != num_classes:
        raise ValueError(f"{name} must be scalar or have num_classes={num_classes} values")
    return out


def _make_class_mask(num_classes: int, target_classes: Optional[Iterable[int]]) -> Tensor:
    """Create a [C] binary mask for constrained classes."""
    mask = torch.zeros(num_classes, dtype=torch.float32)
    if target_classes is None:
        mask.fill_(1.0)
    else:
        for c in target_classes:
            c = int(c)
            if c < 0 or c >= num_classes:
                raise ValueError(f"target class {c} is outside [0, {num_classes - 1}]")
            mask[c] = 1.0
    return mask


def _extract_class_psi(eff: Dict[str, Tensor]) -> Tensor:
    """Support both v2 keys and older compatibility keys."""
    if "class_psi" in eff:
        return eff["class_psi"]
    if "psi" in eff:
        return eff["psi"]
    raise KeyError(
        "chance_corrected_efficacy must return 'class_psi' "
        "or backward-compatible key 'psi'."
    )


def _extract_clamped_class_psi(eff: Dict[str, Tensor], raw_psi: Tensor) -> Tensor:
    if "class_psi_clamped" in eff:
        return eff["class_psi_clamped"]
    if "psi_clamped" in eff:
        return eff["psi_clamped"]
    return raw_psi.clamp(min=-1.0, max=1.0)


class EvidenceEfficacyConstraint(nn.Module):
    """
    Convert evidence efficacy scores into primal-dual constraint violations.

    Supports:
        Classification: evidence_input [B, C], target [B]
        Segmentation:   evidence_input [B, C, H, W], target [B, H, W]

    Important output convention
    ---------------------------
    ``class_violation`` is the MASKED violation. That is intentional:

        class_violation = class_mask * [rho - psi]_+

    ``class_violation_raw`` stores the unmasked values. The backward-compatible
    alias ``violation`` is also the masked version, so older code such as

        (mu * out["violation"]).sum()

    will not accidentally constrain non-target classes.
    """

    def __init__(
        self,
        num_classes: int,
        rho: Union[float, Iterable[float], Tensor] = 0.1,
        target_classes: Optional[Iterable[int]] = None,
        *,
        use_map_constraint: bool = False,
        map_rho: Optional[float] = None,
        ignore_index: Optional[int] = 255,
        eps: float = 1e-6,
        softmin_tau: Optional[float] = None,
        use_prob_input: bool = False,
        use_clamped_psi_for_violation: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.eps = float(eps)
        self.softmin_tau = softmin_tau
        self.use_prob_input = bool(use_prob_input)
        self.use_map_constraint = bool(use_map_constraint)
        self.use_clamped_psi_for_violation = bool(use_clamped_psi_for_violation)

        self.register_buffer("rho", _make_class_vector(rho, self.num_classes, name="rho"))
        self.register_buffer("class_mask", _make_class_mask(self.num_classes, target_classes))

        if self.use_map_constraint and map_rho is None:
            raise ValueError("map_rho must be provided when use_map_constraint=True")
        # map_rho is stored for state_dict consistency, but map-related keys are
        # only emitted when use_map_constraint=True to avoid fake map_psi logs.
        self.register_buffer(
            "map_rho",
            torch.tensor(float(map_rho) if map_rho is not None else 0.0, dtype=torch.float32),
        )

    def forward(
        self,
        evidence_input: Tensor,
        target: Tensor,
        *,
        chance_prior: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Return class-wise and optional map-wise constraint violations.

        The returned dictionary intentionally does not contain a key named
        ``loss``. The final objective must be assembled by primal_dual_loss.py.
        """
        if self.use_prob_input:
            evidence_prob = evidence_input
        else:
            evidence_prob = F.softmax(evidence_input, dim=1)

        eff = chance_corrected_efficacy(
            probs=evidence_prob,
            target=target,
            num_classes=self.num_classes,
            chance_prior=chance_prior,
            ignore_index=self.ignore_index,
            eps=self.eps,
            softmin_tau=self.softmin_tau,
        )

        class_psi_raw = _extract_class_psi(eff)
        class_psi_clamped = _extract_clamped_class_psi(eff, class_psi_raw)

        # Numerical guard:
        # In extremely imbalanced segmentation, some rare-class mini-batches can
        # produce NaN/Inf efficacy values. Never allow them to enter constraint
        # violation or dual-variable update.
        class_psi_raw = torch.nan_to_num(
            class_psi_raw,
            nan=-1.0,
            posinf=1.0,
            neginf=-1.0,
        )
        class_psi_clamped = torch.nan_to_num(
            class_psi_clamped,
            nan=-1.0,
            posinf=1.0,
            neginf=-1.0,
        )

        class_psi_for_violation = (
            class_psi_clamped if self.use_clamped_psi_for_violation else class_psi_raw
        )

        rho = self.rho.to(device=class_psi_raw.device, dtype=class_psi_raw.dtype)
        class_mask = self.class_mask.to(device=class_psi_raw.device, dtype=class_psi_raw.dtype)

        class_violation_raw = F.relu(rho - class_psi_for_violation)
        class_violation = class_violation_raw * class_mask
        class_violation_mean = class_violation.sum() / class_mask.sum().clamp_min(1.0)

        out: Dict[str, Tensor] = {
            "class_violation": class_violation,              # masked, safe default
            "class_violation_raw": class_violation_raw,      # unmasked, analysis only
            "masked_class_violation": class_violation,       # explicit alias
            "class_violation_mean": class_violation_mean,
            "class_psi": class_psi_raw,
            "class_psi_clamped": class_psi_clamped,
            "class_psi_for_violation": class_psi_for_violation,
            "rho": rho,
            "class_mask": class_mask,
            "evidence_prob": evidence_prob,
            "efficacy": eff,
            "map_constraint_enabled": torch.tensor(
                bool(self.use_map_constraint), device=class_psi_raw.device
            ),
        }

        # Backward-compatible aliases. `violation` is intentionally MASKED.
        out["psi"] = class_psi_raw
        out["violation"] = class_violation
        out["masked_violation"] = class_violation
        out["unmasked_violation"] = class_violation_raw

        if self.use_map_constraint:
            if "map_psi" not in eff:
                raise KeyError(
                    "use_map_constraint=True requires chance_corrected_efficacy "
                    "to return 'map_psi'. Update ccel.metrics.efficacy_metrics first."
                )
            map_psi_raw = eff["map_psi"]
            map_psi_raw = torch.nan_to_num(
                map_psi_raw,
                nan=-1.0,
                posinf=1.0,
                neginf=-1.0,
            )
            if not torch.is_tensor(map_psi_raw):
                map_psi_raw = torch.as_tensor(
                    map_psi_raw, device=class_psi_raw.device, dtype=class_psi_raw.dtype
                )
            map_psi_raw = map_psi_raw.to(device=class_psi_raw.device, dtype=class_psi_raw.dtype)
            map_psi_clamped = eff.get("map_psi_clamped", map_psi_raw.clamp(min=-1.0, max=1.0))
            map_psi_clamped = map_psi_clamped.to(device=class_psi_raw.device, dtype=class_psi_raw.dtype)
            map_psi_for_violation = (
                map_psi_clamped if self.use_clamped_psi_for_violation else map_psi_raw
            )
            map_rho = self.map_rho.to(device=class_psi_raw.device, dtype=class_psi_raw.dtype)
            map_violation = F.relu(map_rho - map_psi_for_violation)
            out.update(
                {
                    "map_violation": map_violation,
                    "map_psi": map_psi_raw,
                    "map_psi_clamped": map_psi_clamped,
                    "map_psi_for_violation": map_psi_for_violation,
                    "map_rho": map_rho,
                }
            )

        return out


# Backward-compatible class name. Prefer EvidenceEfficacyConstraint in new code.
EvidenceEfficacyLoss = EvidenceEfficacyConstraint


@torch.no_grad()
def efficacy_constraint_log_dict(prefix: str, out: Dict[str, Tensor]) -> Dict[str, float]:
    """Convert EvidenceEfficacyConstraint output to scalar log values."""
    logs: Dict[str, float] = {}
    psi = out["class_psi"].detach().float().cpu()
    vio = out["class_violation"].detach().float().cpu()
    raw_vio = out["class_violation_raw"].detach().float().cpu()
    mask = out["class_mask"].detach().float().cpu()
    for c in range(psi.numel()):
        logs[f"{prefix}/class{c}_psi"] = float(psi[c])
        logs[f"{prefix}/class{c}_violation"] = float(vio[c])
        logs[f"{prefix}/class{c}_raw_violation"] = float(raw_vio[c])
        logs[f"{prefix}/class{c}_mask"] = float(mask[c])

    logs[f"{prefix}/class_violation_mean"] = float(
        out["class_violation_mean"].detach().float().cpu()
    )

    # Do not log fake map_psi=0 when map constraint is disabled.
    if bool(out.get("map_constraint_enabled", torch.tensor(False)).item()) and "map_psi" in out:
        logs[f"{prefix}/map_psi"] = float(out["map_psi"].detach().float().cpu())
        logs[f"{prefix}/map_violation"] = float(out["map_violation"].detach().float().cpu())
        logs[f"{prefix}/map_rho"] = float(out["map_rho"].detach().float().cpu())
    return logs


# Backward-compatible alias.
efficacy_log_dict = efficacy_constraint_log_dict
