"""
Static prior-correction losses for class-imbalanced baselines.

This file implements static logit-prior correction baselines used to compare
against CCEL-Net. It is NOT the main CCEL-Net training objective.

Implemented baselines:
    1) LogitAdjustedCrossEntropyLoss
       CE(z + tau * log(pi), y) or CE(z - tau * log(pi), y)

    2) BalancedSoftmaxLoss
       CE(z + log(n_c), y), implemented through normalized class prior.

    3) PostHocLogitAdjustment
       Apply the same static correction at evaluation time without computing CE.

Design notes:
    - center_adjustment=True subtracts the mean adjustment over classes. This
      does not change softmax probabilities or CE because adding the same
      constant to every class logit is invariant under softmax. It is used only
      for numerical stability.
    - When center_adjustment=True, the effective adjustment is no longer the raw
      tau * log(pi_c), but centered tau * log(pi_c) - mean_c tau * log(pi_c).
      Logging helpers therefore report both raw_adjustment and adjustment.
    - max_abs_adjustment defaults to None. Clipping creates a stabilized variant
      rather than the standard Logit Adjustment / Balanced Softmax baseline.
      For fair main comparisons, keep max_abs_adjustment=None. If needed,
      report clipped variants separately, e.g. "Logit Adjustment clipped".

Supports:
    classification logits: [B, C], target [B]
    segmentation logits:   [B, C, H, W], target [B, H, W]
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor
PriorLike = Union[Iterable[float], Tensor]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _as_prior(
    values: PriorLike,
    num_classes: int,
    *,
    eps: float,
    name: str,
) -> Tensor:
    """Convert list/tensor to normalized class prior [C]."""
    prior = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if prior.numel() != int(num_classes):
        raise ValueError(f"{name} must contain exactly num_classes={num_classes} values")
    prior = prior.clamp_min(float(eps))
    prior = prior / prior.sum().clamp_min(float(eps))
    return prior


def _as_counts(
    values: PriorLike,
    num_classes: int,
    *,
    eps: float,
    name: str,
) -> Tensor:
    """Convert class counts [C] to normalized class prior [C]."""
    counts = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if counts.numel() != int(num_classes):
        raise ValueError(f"{name} must contain exactly num_classes={num_classes} values")
    if (counts < 0).any():
        raise ValueError(f"{name} must be non-negative")
    counts = counts.clamp_min(float(eps))
    return counts / counts.sum().clamp_min(float(eps))


def _validate_direction(direction: str) -> str:
    direction = str(direction).lower()
    if direction not in {"add", "subtract"}:
        raise ValueError("direction must be 'add' or 'subtract'")
    return direction


def _broadcast_adjustment(adjustment: Tensor, logits: Tensor) -> Tensor:
    """Broadcast [C] adjustment to [B,C] or [B,C,H,W]."""
    if logits.dim() == 2:
        return adjustment.view(1, -1).to(device=logits.device, dtype=logits.dtype)
    if logits.dim() == 4:
        return adjustment.view(1, -1, 1, 1).to(device=logits.device, dtype=logits.dtype)
    raise ValueError(f"logits must be [B,C] or [B,C,H,W], got shape {tuple(logits.shape)}")


def _cross_entropy(logits: Tensor, target: Tensor, ignore_index: Optional[int]) -> Tensor:
    ignore = -100 if ignore_index is None else int(ignore_index)
    return F.cross_entropy(logits, target.long(), ignore_index=ignore)


# -----------------------------------------------------------------------------
# Base static prior adjuster
# -----------------------------------------------------------------------------

class StaticPriorLogitAdjuster(nn.Module):
    """
    Static class-prior logit adjuster.

    Args:
        num_classes:
            Number of classes C.
        class_prior:
            Normalized class prior pi_c or unnormalized positive values that can
            be normalized. For Balanced Softmax, pass class counts through
            BalancedSoftmaxLoss instead.
        tau:
            Temperature multiplier in tau * log(pi_c).
        direction:
            'add' gives z + tau log(pi); 'subtract' gives z - tau log(pi).
        center_adjustment:
            If True, subtract class-wise mean from the adjustment. This does not
            change softmax/CE and is used only for numerical stability.
        max_abs_adjustment:
            Optional clipping threshold. Keep None for standard fair baselines.
            A non-None value should be reported as a clipped/stabilized variant.
    """

    def __init__(
        self,
        num_classes: int,
        class_prior: PriorLike,
        *,
        tau: float = 1.0,
        direction: str = "add",
        eps: float = 1e-12,
        center_adjustment: bool = True,
        max_abs_adjustment: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        self.direction = _validate_direction(direction)
        self.eps = float(eps)
        self.center_adjustment = bool(center_adjustment)
        self.max_abs_adjustment = None if max_abs_adjustment is None else float(max_abs_adjustment)

        prior = _as_prior(class_prior, self.num_classes, eps=self.eps, name="class_prior")
        self.register_buffer("class_prior", prior)

    def raw_adjustment_vector(self) -> Tensor:
        """Return raw tau * log(pi_c), before direction, centering, and clipping."""
        raw = self.tau * torch.log(self.class_prior.clamp_min(self.eps))
        return raw

    def adjustment_vector(self) -> Tensor:
        """
        Return the effective adjustment vector actually applied to logits.

        This includes direction, optional centering, and optional clipping.
        """
        adj = self.raw_adjustment_vector()

        if self.direction == "subtract":
            adj = -adj

        if self.center_adjustment:
            adj = adj - adj.mean()

        if self.max_abs_adjustment is not None:
            adj = adj.clamp(
                min=-float(self.max_abs_adjustment),
                max=float(self.max_abs_adjustment),
            )

        return adj

    def forward(self, logits: Tensor) -> Tensor:
        """Apply static adjustment to logits."""
        adj = self.adjustment_vector()
        return logits + _broadcast_adjustment(adj, logits)

    @torch.no_grad()
    def log_dict(self, prefix: str = "logit_adjustment") -> Dict[str, float]:
        """Return raw and effective adjustment values for logging."""
        raw = self.raw_adjustment_vector().detach().float().cpu()
        effective = self.adjustment_vector().detach().float().cpu()
        prior = self.class_prior.detach().float().cpu()

        logs: Dict[str, float] = {
            f"{prefix}/tau": float(self.tau),
            f"{prefix}/center_adjustment": float(self.center_adjustment),
            f"{prefix}/clipped": float(self.max_abs_adjustment is not None),
            f"{prefix}/max_abs_adjustment": (
                float(self.max_abs_adjustment) if self.max_abs_adjustment is not None else float("nan")
            ),
        }
        for c in range(self.num_classes):
            logs[f"{prefix}/prior_class{c}"] = float(prior[c])
            logs[f"{prefix}/raw_tau_log_prior_class{c}"] = float(raw[c])
            logs[f"{prefix}/effective_centered_adjustment_class{c}"] = float(effective[c])
        return logs


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

class LogitAdjustedCrossEntropyLoss(nn.Module):
    """
    Logit Adjustment baseline.

    Standard main-comparison setting:
        max_abs_adjustment=None

    Clipped/stabilized variant:
        max_abs_adjustment=4.0
    should be reported separately as "Logit Adjustment clipped".
    """

    def __init__(
        self,
        num_classes: int,
        class_prior: PriorLike,
        *,
        tau: float = 1.0,
        direction: str = "add",
        ignore_index: Optional[int] = 255,
        eps: float = 1e-12,
        center_adjustment: bool = True,
        max_abs_adjustment: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.adjuster = StaticPriorLogitAdjuster(
            num_classes=num_classes,
            class_prior=class_prior,
            tau=tau,
            direction=direction,
            eps=eps,
            center_adjustment=center_adjustment,
            max_abs_adjustment=max_abs_adjustment,
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        adjusted_logits = self.adjuster(logits)
        return _cross_entropy(adjusted_logits, target, self.ignore_index)

    def adjusted_logits(self, logits: Tensor) -> Tensor:
        return self.adjuster(logits)

    @torch.no_grad()
    def log_dict(self, prefix: str = "logit_adjustment") -> Dict[str, float]:
        return self.adjuster.log_dict(prefix)


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax baseline.

    Uses CE(z + log(n_c), y). Since adding a constant to all class logits does
    not change softmax, normalized class prior pi_c is sufficient.

    Standard main-comparison setting:
        max_abs_adjustment=None

    If max_abs_adjustment is set, report it as "Balanced Softmax clipped".
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: PriorLike,
        *,
        ignore_index: Optional[int] = 255,
        eps: float = 1e-12,
        center_adjustment: bool = True,
        max_abs_adjustment: Optional[float] = None,
    ) -> None:
        super().__init__()
        prior = _as_counts(class_counts, num_classes, eps=eps, name="class_counts")
        self.ignore_index = ignore_index
        self.adjuster = StaticPriorLogitAdjuster(
            num_classes=num_classes,
            class_prior=prior,
            tau=1.0,
            direction="add",
            eps=eps,
            center_adjustment=center_adjustment,
            max_abs_adjustment=max_abs_adjustment,
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        adjusted_logits = self.adjuster(logits)
        return _cross_entropy(adjusted_logits, target, self.ignore_index)

    def adjusted_logits(self, logits: Tensor) -> Tensor:
        return self.adjuster(logits)

    @torch.no_grad()
    def log_dict(self, prefix: str = "balanced_softmax") -> Dict[str, float]:
        return self.adjuster.log_dict(prefix)


class PostHocLogitAdjustment(nn.Module):
    """Apply static logit adjustment without computing a loss."""

    def __init__(
        self,
        num_classes: int,
        class_prior: PriorLike,
        *,
        tau: float = 1.0,
        direction: str = "subtract",
        eps: float = 1e-12,
        center_adjustment: bool = True,
        max_abs_adjustment: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.adjuster = StaticPriorLogitAdjuster(
            num_classes=num_classes,
            class_prior=class_prior,
            tau=tau,
            direction=direction,
            eps=eps,
            center_adjustment=center_adjustment,
            max_abs_adjustment=max_abs_adjustment,
        )

    def forward(self, logits: Tensor) -> Tensor:
        return self.adjuster(logits)

    @torch.no_grad()
    def log_dict(self, prefix: str = "posthoc_logit_adjustment") -> Dict[str, float]:
        return self.adjuster.log_dict(prefix)


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def apply_logit_adjustment(
    logits: Tensor,
    class_prior: PriorLike,
    *,
    tau: float = 1.0,
    direction: str = "add",
    eps: float = 1e-12,
    center_adjustment: bool = True,
    max_abs_adjustment: Optional[float] = None,
) -> Tensor:
    """Functional API for one-off static logit adjustment."""
    num_classes = logits.size(1)
    adjuster = StaticPriorLogitAdjuster(
        num_classes=num_classes,
        class_prior=class_prior,
        tau=tau,
        direction=direction,
        eps=eps,
        center_adjustment=center_adjustment,
        max_abs_adjustment=max_abs_adjustment,
    ).to(device=logits.device)
    return adjuster(logits)


__all__ = [
    "StaticPriorLogitAdjuster",
    "LogitAdjustedCrossEntropyLoss",
    "BalancedSoftmaxLoss",
    "PostHocLogitAdjustment",
    "apply_logit_adjustment",
]
