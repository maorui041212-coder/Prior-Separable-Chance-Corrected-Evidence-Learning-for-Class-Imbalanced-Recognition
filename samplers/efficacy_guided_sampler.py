"""Helpers for efficacy-guided rebalancing."""
from __future__ import annotations

from typing import Optional

import torch


def efficacy_guided_class_prob(
    psi: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-6,
    min_gap: float = 0.0,
    max_gap: float = 1.0,
    class_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    P(c) proportional to (1 - psi_c)^alpha.

    psi should be class-wise efficacy. For stable sampling, the learning gap is
    clamped into [min_gap, max_gap].
    """
    psi = psi.detach().float()
    gap = (1.0 - psi).clamp(min=float(min_gap), max=float(max_gap))

    if class_mask is not None:
        class_mask = class_mask.to(device=gap.device, dtype=gap.dtype)
        gap = gap * class_mask

    weights = (gap + eps).pow(float(alpha))

    if class_mask is not None:
        weights = weights * class_mask

    if weights.sum() <= eps:
        if class_mask is not None and class_mask.sum() > 0:
            weights = class_mask / class_mask.sum().clamp_min(eps)
        else:
            weights = torch.ones_like(weights) / weights.numel()

    return weights / weights.sum().clamp_min(eps)


def labels_to_sample_weights(
    labels: torch.Tensor,
    class_prob: torch.Tensor,
    ignore_index: Optional[int] = None,
    invalid_weight: float = 0.0,
) -> torch.Tensor:
    """Convert image-level labels [N] to per-sample weights [N]."""
    labels = labels.long()
    class_prob = class_prob.to(device=labels.device)

    valid = (labels >= 0) & (labels < class_prob.numel())
    if ignore_index is not None:
        valid = valid & (labels != int(ignore_index))

    weights = torch.full(
        labels.shape,
        float(invalid_weight),
        device=labels.device,
        dtype=class_prob.dtype,
    )
    weights[valid] = class_prob[labels[valid]]
    return weights


def segmentation_mask_to_sample_weight(
    mask: torch.Tensor,
    class_prob: torch.Tensor,
    ignore_index: Optional[int] = 255,
    mode: str = "max",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Convert one segmentation mask/patch [H,W] to a scalar sampling weight.

    mode="max":
        Uses the largest class probability among classes present in the patch.
        Recommended for small-object segmentation.

    mode="mean":
        Uses histogram-weighted average of class probabilities.
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be [H,W], got {tuple(mask.shape)}")

    class_prob = class_prob.float()
    mask = mask.to(device=class_prob.device).long()

    valid = (mask >= 0) & (mask < class_prob.numel())
    if ignore_index is not None:
        valid = valid & (mask != int(ignore_index))

    labels = mask[valid]
    if labels.numel() == 0:
        return torch.tensor(float(eps), device=class_prob.device, dtype=class_prob.dtype)

    present = torch.unique(labels)

    if mode == "max":
        return class_prob[present].max().clamp_min(eps)

    if mode == "mean":
        hist = torch.bincount(labels, minlength=class_prob.numel()).to(
            device=class_prob.device,
            dtype=class_prob.dtype,
        )
        hist = hist / hist.sum().clamp_min(eps)
        return (hist * class_prob).sum().clamp_min(eps)

    raise ValueError("mode must be 'max' or 'mean'")