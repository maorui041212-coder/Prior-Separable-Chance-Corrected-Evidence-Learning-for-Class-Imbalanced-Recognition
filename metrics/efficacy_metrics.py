"""
Chance-corrected evidence efficacy metrics for CCEL-Net.

Design boundary
---------------
This file owns metric/statistic computation only. It should NOT know about rho,
mu, CE, lambda weights, or primal-dual optimization.

It provides:
    1) differentiable soft confusion statistics from evidence probabilities;
    2) class-wise chance-corrected efficacy psi_c;
    3) map-wise chance-corrected efficacy psi_map;
    4) raw and clamped psi values for downstream constraint adapters;
    5) reliability masks for extremely imbalanced mini-batches;
    6) an EMA meter for stable dual-variable updates.

Supported shapes:
    Classification: probs [B, C],       target [B]
    Segmentation:   probs [B, C, H, W], target [B, H, W]
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Basic tensor utilities
# -----------------------------------------------------------------------------

def _flatten_probs_and_target(
    probs: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Return probs [N, C] and target [N] after optional ignore masking.

    This function only removes ignore_index. Out-of-range labels are filtered in
    soft_confusion_matrix so that accidental labels such as -1 or 999 do not get
    silently clamped into a valid class.
    """
    if probs.dim() == 2:
        flat_probs = probs
        flat_target = target.reshape(-1)
        if flat_probs.size(0) != flat_target.numel():
            raise ValueError(
                f"For classification probs [B,C], target must have B elements; "
                f"got probs={tuple(probs.shape)}, target={tuple(target.shape)}"
            )
    elif probs.dim() == 4:
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.size(1))
        flat_target = target.reshape(-1)
        if flat_probs.size(0) != flat_target.numel():
            raise ValueError(
                f"For segmentation probs [B,C,H,W], target must be [B,H,W]; "
                f"got probs={tuple(probs.shape)}, target={tuple(target.shape)}"
            )
    else:
        raise ValueError(
            f"probs must be [B,C] or [B,C,H,W], got shape {tuple(probs.shape)}"
        )

    flat_target = flat_target.long()
    if ignore_index is not None:
        valid = flat_target != int(ignore_index)
        flat_probs = flat_probs[valid]
        flat_target = flat_target[valid]

    return flat_probs, flat_target


def smooth_min(a: Tensor, b: Tensor, tau: float = 0.05) -> Tensor:
    """Differentiable approximation to min(a, b)."""
    if tau <= 0:
        return torch.minimum(a, b)
    stacked = torch.stack([-a / tau, -b / tau], dim=0)
    return -tau * torch.logsumexp(stacked, dim=0)


def _safe_normalize_prior(prior: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize a prior vector safely."""
    prior = prior.clamp_min(0.0)
    return prior / prior.sum().clamp_min(eps)


def _zero_stats(num_classes: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Tensor]:
    zc = torch.zeros(num_classes, device=device, dtype=dtype)
    return {
        "soft_confusion": torch.zeros(num_classes, num_classes, device=device, dtype=dtype),
        "true_prior": zc.clone(),
        "pred_prior": zc.clone(),
        "class_count": zc.clone(),
        "valid_count": torch.zeros((), device=device, dtype=dtype),
    }


# -----------------------------------------------------------------------------
# Soft confusion statistics
# -----------------------------------------------------------------------------

def soft_confusion_matrix(
    probs: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    normalize_by_valid: bool = True,
) -> Dict[str, Tensor]:
    """
    Compute soft confusion matrix from evidence probabilities.

        M[a, b] = 1/N * sum_i 1[y_i = a] * p_i(b)

    Returns:
        soft_confusion: [C, C]
        true_prior:     [C], pi_a = mean_i 1[y_i=a]
        pred_prior:     [C], r_b  = mean_i p_i(b)
        class_count:    [C], number of valid samples/pixels for each true class
        valid_count:    scalar tensor, number of valid samples/pixels
    """
    if probs.size(1) != num_classes:
        raise ValueError(
            f"probs channel dimension must equal num_classes={num_classes}, "
            f"got probs.shape={tuple(probs.shape)}"
        )

    flat_probs, flat_target = _flatten_probs_and_target(probs, target, ignore_index)
    device = probs.device
    dtype = probs.dtype

    if flat_probs.numel() == 0:
        return _zero_stats(num_classes, device, dtype)

    # Do NOT clamp out-of-range labels into valid classes; filter them out.
    valid_label = (flat_target >= 0) & (flat_target < int(num_classes))
    flat_probs = flat_probs[valid_label]
    flat_target = flat_target[valid_label]

    if flat_probs.numel() == 0:
        return _zero_stats(num_classes, device, dtype)

    valid_count = torch.tensor(float(flat_probs.size(0)), device=device, dtype=dtype)
    one_hot = F.one_hot(flat_target, num_classes=num_classes).to(dtype=dtype, device=device)

    raw_confusion = one_hot.transpose(0, 1) @ flat_probs
    class_count = one_hot.sum(dim=0)

    if normalize_by_valid:
        denom = valid_count.clamp_min(1.0)
    else:
        # Mostly for compatibility. For segmentation with ignore_index, the
        # normalized-by-valid version is usually the correct one.
        denom = torch.tensor(float(max(1, target.numel())), device=device, dtype=dtype)

    soft_confusion = raw_confusion / denom
    true_prior = class_count / denom
    pred_prior = flat_probs.sum(dim=0) / denom

    return {
        "soft_confusion": soft_confusion,
        "true_prior": true_prior,
        "pred_prior": pred_prior,
        "class_count": class_count,
        "valid_count": valid_count,
    }


# -----------------------------------------------------------------------------
# Efficacy from statistics
# -----------------------------------------------------------------------------

def efficacy_from_confusion(
    soft_confusion: Tensor,
    true_prior: Tensor,
    pred_prior: Tensor,
    chance_prior: Optional[Tensor] = None,
    eps: float = 1e-6,
    softmin_tau: Optional[float] = None,
) -> Dict[str, Tensor]:
    """
    Compute class-wise and map-wise chance-corrected efficacy.

    Class-wise:
        psi_c = (M_cc - pi_tilde_c * r_c) / (U_c - pi_tilde_c * r_c + eps)
        U_c   = min(pi_tilde_c, r_c)

    Map-wise:
        A       = sum_c M_cc
        A0      = sum_c pi_tilde_c * r_c
        U_map   = sum_c min(pi_tilde_c, r_c)
        psi_map = (A - A0) / (U_map - A0 + eps)

    This function always returns raw psi and clamped psi separately. It never
    overwrites raw psi, because constraint/loss code should decide whether to
    use raw or clamped values.
    """
    if soft_confusion.dim() != 2 or soft_confusion.size(0) != soft_confusion.size(1):
        raise ValueError("soft_confusion must be [C, C]")

    num_classes = soft_confusion.size(0)
    dtype = soft_confusion.dtype
    device = soft_confusion.device

    if chance_prior is None:
        pi_tilde = _safe_normalize_prior(true_prior.to(device=device, dtype=dtype), eps=eps)
    else:
        pi_tilde = chance_prior.to(device=device, dtype=dtype).reshape(-1)
        if pi_tilde.numel() != num_classes:
            raise ValueError(f"chance_prior must be a [C] tensor with C={num_classes}")
        pi_tilde = _safe_normalize_prior(pi_tilde, eps=eps)

    pred_prior = pred_prior.to(device=device, dtype=dtype)
    true_prior = true_prior.to(device=device, dtype=dtype)

    diag = torch.diag(soft_confusion)
    class_chance = pi_tilde * pred_prior

    if softmin_tau is None:
        class_upper = torch.minimum(pi_tilde, pred_prior)
    else:
        class_upper = smooth_min(pi_tilde, pred_prior, tau=float(softmin_tau))

    class_denom = (class_upper - class_chance).clamp_min(eps)
    class_psi = (diag - class_chance) / class_denom
    class_psi_clamped = class_psi.clamp(min=-1.0, max=1.0)

    map_correct_mass = diag.sum()
    map_chance = class_chance.sum()
    map_upper = class_upper.sum()
    map_denom = (map_upper - map_chance).clamp_min(eps)
    map_psi = (map_correct_mass - map_chance) / map_denom
    map_psi_clamped = map_psi.clamp(min=-1.0, max=1.0)

    return {
        # Preferred keys.
        "class_psi": class_psi,
        "class_psi_clamped": class_psi_clamped,
        "map_psi": map_psi,
        "map_psi_clamped": map_psi_clamped,
        # Backward-compatible aliases.
        "psi": class_psi,
        "psi_clamped": class_psi_clamped,
        # Statistics.
        "soft_confusion": soft_confusion,
        "true_prior": true_prior,
        "evidence_pred_prior": pred_prior,
        "pred_prior": pred_prior,
        "chance_prior": pi_tilde,
        "class_chance_baseline": class_chance,
        "class_upper_bound": class_upper,
        "class_denom": class_denom,
        "class_correct_mass": diag,
        "map_correct_mass": map_correct_mass,
        "map_chance_baseline": map_chance,
        "map_upper_bound": map_upper,
        "map_denom": map_denom,
    }


# -----------------------------------------------------------------------------
# Public functional API
# -----------------------------------------------------------------------------

def chance_corrected_efficacy(
    probs: Tensor,
    target: Tensor,
    num_classes: int,
    chance_prior: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
    softmin_tau: Optional[float] = None,
    min_class_count: int = 1,
    min_valid_count: int = 1,
    clamp_psi: Optional[bool] = None,
) -> Dict[str, Tensor]:
    """
    Compute class-wise and map-wise evidence efficacy from one mini-batch.

    Notes:
        - ``class_psi`` and ``map_psi`` are raw values.
        - ``class_psi_clamped`` and ``map_psi_clamped`` are always provided.
        - ``clamp_psi`` is kept only for backward compatibility and is ignored;
          downstream modules should explicitly choose raw or clamped values.
    """
    stats = soft_confusion_matrix(
        probs=probs,
        target=target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    eff = efficacy_from_confusion(
        soft_confusion=stats["soft_confusion"],
        true_prior=stats["true_prior"],
        pred_prior=stats["pred_prior"],
        chance_prior=chance_prior,
        eps=eps,
        softmin_tau=softmin_tau,
    )

    class_count = stats["class_count"]
    valid_count = stats["valid_count"]
    class_reliable_mask = class_count >= float(min_class_count)
    map_reliable = valid_count >= float(min_valid_count)

    eff.update(
        {
            "class_count": class_count,
            "valid_count": valid_count,
            "class_reliable_mask": class_reliable_mask,
            "map_reliable": map_reliable,
        }
    )
    return eff


# -----------------------------------------------------------------------------
# EMA meter for stable dual-variable updates
# -----------------------------------------------------------------------------

class ChanceCorrectedEfficacyMeter(nn.Module):
    """
    EMA estimator of chance-corrected efficacy.

    Why this exists:
        In extremely imbalanced segmentation, a mini-batch may contain very few
        minority pixels, or no minority pixels at all. Updating dual variables
        mu_c from such noisy psi_c can make mu explode or oscillate. This meter
        maintains EMA statistics across iterations and provides smoothed psi for
        dual-variable updates.

    Recommended use:
        - use batch psi from chance_corrected_efficacy/EvidenceEfficacyConstraint
          for differentiable constraint loss;
        - use EMA psi and reliability masks from this meter for update_dual().
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = 255,
        momentum: float = 0.95,
        eps: float = 1e-6,
        softmin_tau: Optional[float] = None,
        min_class_count: int = 32,
        min_valid_count: int = 256,
        warmup_updates: int = 10,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.softmin_tau = softmin_tau
        self.min_class_count = int(min_class_count)
        self.min_valid_count = int(min_valid_count)
        self.warmup_updates = int(warmup_updates)

        self.register_buffer("ema_soft_confusion", torch.zeros(num_classes, num_classes))
        self.register_buffer("ema_true_prior", torch.zeros(num_classes))
        self.register_buffer("ema_pred_prior", torch.zeros(num_classes))
        self.register_buffer("ema_class_count", torch.zeros(num_classes))
        self.register_buffer("ema_valid_count", torch.zeros(()))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def reset(self) -> None:
        self.ema_soft_confusion.zero_()
        self.ema_true_prior.zero_()
        self.ema_pred_prior.zero_()
        self.ema_class_count.zero_()
        self.ema_valid_count.zero_()
        self.num_updates.zero_()
        self.initialized.fill_(False)

    @torch.no_grad()
    def _ensure_buffer_device_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move floating EMA buffers to the current training device/dtype."""
        for name in [
            "ema_soft_confusion",
            "ema_true_prior",
            "ema_pred_prior",
            "ema_class_count",
            "ema_valid_count",
        ]:
            buf = getattr(self, name)
            if buf.device != device or buf.dtype != dtype:
                # Re-assigning to an existing buffer name keeps it registered in nn.Module.
                setattr(self, name, buf.to(device=device, dtype=dtype))
        for name in ["num_updates", "initialized"]:
            buf = getattr(self, name)
            if buf.device != device:
                setattr(self, name, buf.to(device=device))

    @torch.no_grad()
    def update(self, probs: Tensor, target: Tensor) -> Dict[str, Tensor]:
        """Update EMA statistics from the current mini-batch."""
        self._ensure_buffer_device_dtype(probs.device, probs.dtype)

        stats = soft_confusion_matrix(
            probs=probs.detach(),
            target=target.detach(),
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        if not bool(self.initialized.item()):
            self.ema_soft_confusion.copy_(stats["soft_confusion"])
            self.ema_true_prior.copy_(stats["true_prior"])
            self.ema_pred_prior.copy_(stats["pred_prior"])
            self.ema_class_count.copy_(stats["class_count"])
            self.ema_valid_count.copy_(stats["valid_count"])
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.ema_soft_confusion.mul_(m).add_(stats["soft_confusion"], alpha=1.0 - m)
            self.ema_true_prior.mul_(m).add_(stats["true_prior"], alpha=1.0 - m)
            self.ema_pred_prior.mul_(m).add_(stats["pred_prior"], alpha=1.0 - m)
            self.ema_class_count.mul_(m).add_(stats["class_count"], alpha=1.0 - m)
            self.ema_valid_count.mul_(m).add_(stats["valid_count"], alpha=1.0 - m)

        self.num_updates.add_(1)
        return stats

    @torch.no_grad()
    def compute(self, chance_prior: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute efficacy from EMA statistics.

        If chance_prior is None, EMA true prior is used as pi_tilde. This is the
        safest default for dual-variable updates because it avoids mixing current
        mini-batch priors with EMA confusion statistics.
        """
        eff = efficacy_from_confusion(
            soft_confusion=self.ema_soft_confusion,
            true_prior=self.ema_true_prior,
            pred_prior=self.ema_pred_prior,
            chance_prior=chance_prior,
            eps=self.eps,
            softmin_tau=self.softmin_tau,
        )

        class_reliable_mask = self.ema_class_count >= float(self.min_class_count)
        map_reliable = self.ema_valid_count >= float(self.min_valid_count)
        warm = self.num_updates >= int(self.warmup_updates)

        eff.update(
            {
                "class_count": self.ema_class_count.clone(),
                "valid_count": self.ema_valid_count.clone(),
                "class_reliable_mask": class_reliable_mask & warm,
                "map_reliable": map_reliable & warm,
                "num_updates": self.num_updates.clone(),
                "is_warm": warm.clone(),
            }
        )
        return eff

    def forward(
        self,
        probs: Tensor,
        target: Tensor,
        chance_prior: Optional[Tensor] = None,
        *,
        chance_prior_for_dual: Optional[Tensor] = None,
        update: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Return both batch efficacy and EMA efficacy.

        Args:
            chance_prior:
                pi_tilde used for differentiable batch efficacy.
            chance_prior_for_dual:
                pi_tilde used for EMA efficacy and dual update. If None, EMA
                true prior is used. Prefer None or a global prior; avoid passing
                the current batch prior here for extremely imbalanced data.
        """
        batch_eff = chance_corrected_efficacy(
            probs=probs,
            target=target,
            num_classes=self.num_classes,
            chance_prior=chance_prior,
            ignore_index=self.ignore_index,
            eps=self.eps,
            softmin_tau=self.softmin_tau,
            min_class_count=self.min_class_count,
            min_valid_count=self.min_valid_count,
        )

        if update:
            self.update(probs=probs, target=target)
        else:
            self._ensure_buffer_device_dtype(probs.device, probs.dtype)

        with torch.no_grad():
            ema_eff = self.compute(chance_prior=chance_prior_for_dual)

        out: Dict[str, Tensor] = {}
        for k, v in batch_eff.items():
            out[f"batch_{k}"] = v
        for k, v in ema_eff.items():
            out[f"ema_{k}"] = v

        # Convenient aliases.
        out["psi"] = batch_eff["class_psi"]
        out["psi_clamped"] = batch_eff["class_psi_clamped"]
        out["class_psi"] = batch_eff["class_psi"]
        out["class_psi_clamped"] = batch_eff["class_psi_clamped"]
        out["map_psi"] = batch_eff["map_psi"]
        out["map_psi_clamped"] = batch_eff["map_psi_clamped"]

        # Use these for dual update.
        out["psi_for_dual"] = ema_eff["class_psi"]
        out["psi_clamped_for_dual"] = ema_eff["class_psi_clamped"]
        out["map_psi_for_dual"] = ema_eff["map_psi"]
        out["map_psi_clamped_for_dual"] = ema_eff["map_psi_clamped"]
        out["dual_update_mask"] = ema_eff["class_reliable_mask"]
        out["map_dual_update_mask"] = ema_eff["map_reliable"]
        return out


# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------

@torch.no_grad()
def efficacy_to_log_dict(prefix: str, efficacy: Dict[str, Tensor]) -> Dict[str, float]:
    """Convert tensors from efficacy dict into scalar log values."""
    out: Dict[str, float] = {}

    psi = efficacy.get("class_psi", efficacy.get("psi", None))
    psi_clamped = efficacy.get("class_psi_clamped", efficacy.get("psi_clamped", None))
    r = efficacy.get("evidence_pred_prior", efficacy.get("pred_prior", None))
    pi = efficacy.get("chance_prior", None)
    class_count = efficacy.get("class_count", None)

    if psi is not None:
        psi_cpu = psi.detach().cpu()
        for c in range(psi_cpu.numel()):
            out[f"{prefix}/psi_class{c}"] = float(psi_cpu[c])

    if psi_clamped is not None:
        psi_cpu = psi_clamped.detach().cpu()
        for c in range(psi_cpu.numel()):
            out[f"{prefix}/psi_clamped_class{c}"] = float(psi_cpu[c])

    if "map_psi" in efficacy:
        out[f"{prefix}/psi_map"] = float(efficacy["map_psi"].detach().cpu())
    if "map_psi_clamped" in efficacy:
        out[f"{prefix}/psi_clamped_map"] = float(efficacy["map_psi_clamped"].detach().cpu())

    if r is not None:
        r_cpu = r.detach().cpu()
        for c in range(r_cpu.numel()):
            out[f"{prefix}/pred_prior_class{c}"] = float(r_cpu[c])

    if pi is not None:
        pi_cpu = pi.detach().cpu()
        for c in range(pi_cpu.numel()):
            out[f"{prefix}/chance_prior_class{c}"] = float(pi_cpu[c])

    if class_count is not None:
        cnt_cpu = class_count.detach().cpu()
        for c in range(cnt_cpu.numel()):
            out[f"{prefix}/count_class{c}"] = float(cnt_cpu[c])

    if "valid_count" in efficacy:
        out[f"{prefix}/valid_count"] = float(efficacy["valid_count"].detach().cpu())

    return out
