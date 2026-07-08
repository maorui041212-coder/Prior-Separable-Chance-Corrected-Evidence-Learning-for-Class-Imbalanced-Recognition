"""Primal-dual efficacy objective for CCEL-Net.

This file owns the final training objective and dual-variable updates:

    L = CE(b + e, y) + sum_c mu_c * [rho_c - psi_c]_+
        + mu_map * [rho_map - psi_map]_+

It must NOT reimplement soft confusion matrix or chance-corrected efficacy.
Those formulas live in ``ccel.metrics.efficacy_metrics``. It must also NOT
recompute violation logic directly; violation construction lives in
``ccel.losses.evidence_efficacy_loss``.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from ccel.losses.evidence_efficacy_loss import EvidenceEfficacyConstraint

Tensor = torch.Tensor


class PrimalDualEfficacyLoss(nn.Module):
    """CE + dynamic dual variables times evidence efficacy violations.

    ``mu`` and ``mu_map`` are buffers, not learnable parameters. They are updated
    by projected dual ascent, not by back-propagation.
    """

    def __init__(
        self,
        num_classes: int,
        rho: Union[float, Iterable[float], Tensor] = 0.1,
        target_classes: Optional[Iterable[int]] = None,
        *,
        eta_mu: float = 0.01,
        mu_max: float = 10.0,
        ignore_index: Optional[int] = 255,
        eps: float = 1e-6,
        softmin_tau: Optional[float] = None,
        ce_weight: Optional[Tensor] = None,
        use_map_constraint: bool = False,
        map_rho: Optional[float] = None,
        eta_mu_map: Optional[float] = None,
        mu_map_max: Optional[float] = None,
        use_clamped_psi_for_violation: bool = True,
        auto_update_mu: bool = True,
        update_mu_on_eval: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.eta_mu = float(eta_mu)
        self.mu_max = float(mu_max)
        self.ignore_index = ignore_index
        self.auto_update_mu = bool(auto_update_mu)
        self.update_mu_on_eval = bool(update_mu_on_eval)
        self.use_map_constraint = bool(use_map_constraint)
        self.eta_mu_map = float(eta_mu if eta_mu_map is None else eta_mu_map)
        self.mu_map_max = float(mu_max if mu_map_max is None else mu_map_max)

        self.constraint = EvidenceEfficacyConstraint(
            num_classes=num_classes,
            rho=rho,
            target_classes=target_classes,
            use_map_constraint=use_map_constraint,
            map_rho=map_rho,
            ignore_index=ignore_index,
            eps=eps,
            softmin_tau=softmin_tau,
            use_prob_input=False,
            use_clamped_psi_for_violation=use_clamped_psi_for_violation,
        )

        self.register_buffer("mu", torch.zeros(num_classes, dtype=torch.float32))
        self.register_buffer("mu_map", torch.zeros((), dtype=torch.float32))

        if ce_weight is not None:
            ce_weight = torch.as_tensor(ce_weight, dtype=torch.float32).reshape(-1)
            if ce_weight.numel() != num_classes:
                raise ValueError("ce_weight must have shape [num_classes]")
            self.register_buffer("ce_weight", ce_weight)
        else:
            self.ce_weight = None

    def forward(
        self,
        outputs: Dict[str, Tensor],
        target: Tensor,
        *,
        chance_prior: Optional[Tensor] = None,
        use_constraint: bool = True,
        auto_update_mu: Optional[bool] = None,
        dual_class_psi: Optional[Tensor] = None,
        dual_update_mask: Optional[Tensor] = None,
        dual_map_psi: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute objective and optionally update dual variables.

        Args:
            outputs:
                Must contain ``logits`` and ``evidence_logits``.
            target:
                Ground-truth labels.
            chance_prior:
                Optional pi_tilde [C].
            dual_class_psi:
                Optional stable/EMA psi used only for dual update. If None,
                the current batch ``class_psi_for_violation`` is used.
            dual_update_mask:
                Optional [C] mask used only for dual update, e.g. reliability
                mask from an EMA efficacy meter.
            dual_map_psi:
                Optional stable/EMA map psi used only for map dual update.
        """
        logits = outputs["logits"]
        evidence_logits = outputs["evidence_logits"]

        ce = F.cross_entropy(
            logits.float(),
            target.long(),
            weight=self.ce_weight,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )

        constraint_out = self.constraint(
            evidence_logits.float(),
            target,
            chance_prior=chance_prior,
        )

        mu_before = self.mu.to(device=logits.device, dtype=logits.dtype).clone()
        mu_map_before = self.mu_map.to(device=logits.device, dtype=logits.dtype).clone()
        class_mask = constraint_out["class_mask"].to(device=logits.device, dtype=logits.dtype)
        class_violation = constraint_out["class_violation"].to(device=logits.device, dtype=logits.dtype)
        class_violation = torch.nan_to_num(
            class_violation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        denom = class_mask.sum().clamp_min(1.0)

        if use_constraint:
            class_constraint = (mu_before.detach() * class_violation).sum() / denom
            if self.use_map_constraint:
                map_constraint = mu_map_before.detach() * constraint_out["map_violation"].to(
                    device=logits.device, dtype=logits.dtype
                )
            else:
                map_constraint = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            class_constraint = torch.zeros((), device=logits.device, dtype=logits.dtype)
            map_constraint = torch.zeros((), device=logits.device, dtype=logits.dtype)

        total = ce + class_constraint + map_constraint

        should_update = self.auto_update_mu if auto_update_mu is None else bool(auto_update_mu)
        should_update = bool(should_update and use_constraint and (self.training or self.update_mu_on_eval))

        if should_update:
            if dual_class_psi is None:
                dual_class_psi = constraint_out["class_psi_for_violation"].detach()
            if dual_update_mask is None:
                # Prefer reliability mask from metrics if available; otherwise use target class mask.
                eff = constraint_out.get("efficacy", {})
                dual_update_mask = eff.get("dual_update_mask", None)
                if dual_update_mask is None:
                    dual_update_mask = eff.get("class_reliable_mask", None)
                if dual_update_mask is None:
                    dual_update_mask = class_mask.detach()
                else:
                    dual_update_mask = dual_update_mask.to(device=class_mask.device, dtype=class_mask.dtype) * class_mask.detach()

            if self.use_map_constraint:
                if dual_map_psi is None:
                    dual_map_psi = constraint_out["map_psi_for_violation"].detach()
                self.update_dual(
                    dual_class_psi=dual_class_psi,
                    dual_update_mask=dual_update_mask,
                    dual_map_psi=dual_map_psi,
                )
            else:
                self.update_dual(
                    dual_class_psi=dual_class_psi,
                    dual_update_mask=dual_update_mask,
                )

        return {
            "loss": total,
            "ce_loss": ce.detach(),
            "constraint_loss": (class_constraint + map_constraint).detach(),
            "class_constraint_loss": class_constraint.detach(),
            "map_constraint_loss": map_constraint.detach(),
            "constraint": constraint_out,
            "psi": constraint_out["class_psi"],
            "class_psi": constraint_out["class_psi"],
            "class_violation": constraint_out["class_violation"].detach(),
            "mu": self.mu.detach().clone(),
            "mu_before": mu_before.detach(),
            "mu_map": self.mu_map.detach().clone(),
            "mu_map_before": mu_map_before.detach(),
        }

    @torch.no_grad()
    def update_dual(
        self,
        *,
        dual_class_psi: Tensor,
        dual_update_mask: Optional[Tensor] = None,
        dual_map_psi: Optional[Tensor] = None,
    ) -> None:
        """Projected dual ascent.

        Class update:
            mu_c <- clip(mu_c + eta_mu * (rho_c - psi_c) * mask_c, 0, mu_max)

        Map update, if enabled:
            mu_map <- clip(mu_map + eta_mu_map * (rho_map - psi_map), 0, mu_map_max)
        """
        psi = dual_class_psi.detach().to(device=self.mu.device, dtype=self.mu.dtype).reshape(-1)
        psi = torch.nan_to_num(
            psi,
            nan=-1.0,
            posinf=1.0,
            neginf=-1.0,
        )
        if psi.numel() != self.num_classes:
            raise ValueError(f"dual_class_psi must have shape [{self.num_classes}]")

        rho = self.constraint.rho.to(device=self.mu.device, dtype=self.mu.dtype)
        class_mask = self.constraint.class_mask.to(device=self.mu.device, dtype=self.mu.dtype)
        if dual_update_mask is None:
            update_mask = class_mask
        else:
            update_mask = dual_update_mask.detach().to(device=self.mu.device, dtype=self.mu.dtype).reshape(-1)
            if update_mask.numel() != self.num_classes:
                raise ValueError(f"dual_update_mask must have shape [{self.num_classes}]")
            update_mask = update_mask * class_mask

        self.mu.add_(self.eta_mu * (rho - psi) * update_mask)
        self.mu.clamp_(min=0.0, max=self.mu_max)
        self.mu.nan_to_num_(
            nan=0.0,
            posinf=float(self.mu_max),
            neginf=0.0,
        )

        if self.use_map_constraint:
            if dual_map_psi is None:
                raise ValueError("dual_map_psi must be provided when use_map_constraint=True")
            map_psi = dual_map_psi.detach().to(device=self.mu_map.device, dtype=self.mu_map.dtype)
            map_psi = torch.nan_to_num(
                map_psi,
                nan=-1.0,
                posinf=1.0,
                neginf=-1.0,
            )
            map_rho = self.constraint.map_rho.to(device=self.mu_map.device, dtype=self.mu_map.dtype)
            self.mu_map.add_(self.eta_mu_map * (map_rho - map_psi))
            self.mu_map.clamp_(min=0.0, max=self.mu_map_max)
            self.mu_map.nan_to_num_(
                nan=0.0,
                posinf=float(self.mu_map_max),
                neginf=0.0,
            )

    @torch.no_grad()
    def reset_dual(self) -> None:
        self.mu.zero_()
        self.mu_map.zero_()
