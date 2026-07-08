"""Class-frequency-based sampling helpers for CCEL-Net baselines.

This file implements traditional frequency-based rebalancing. It is used as a
baseline/ablation against efficacy-guided sampling.

Difference from efficacy_guided_sampler.py:
    efficacy-guided:
        P(c) proportional to (1 - psi_c)^alpha

    class-frequency:
        P(c) proportional to 1 / freq_c^alpha

For classification, this file converts image labels into per-sample weights.
For segmentation, this file converts masks/patches into patch-level weights.
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch
from torch.utils.data import WeightedRandomSampler

Tensor = torch.Tensor


def _make_class_mask(
    num_classes: int,
    target_classes: Optional[Iterable[int]] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a [C] mask. If target_classes is None, all classes are enabled."""
    mask = torch.zeros(num_classes, device=device, dtype=dtype)
    if target_classes is None:
        mask.fill_(1.0)
    else:
        for c in target_classes:
            c = int(c)
            if c < 0 or c >= num_classes:
                raise ValueError(f"target class {c} is outside [0, {num_classes - 1}]")
            mask[c] = 1.0
    return mask


def counts_to_inverse_frequency_prob(
    class_counts: Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-6,
    target_classes: Optional[Iterable[int]] = None,
    normalize: bool = True,
) -> Tensor:
    """
    Convert class counts to inverse-frequency class probabilities.

    Formula:
        P(c) proportional to 1 / (count_c + eps)^alpha

    Args:
        class_counts:
            [C] class counts. For classification, image counts.
            For segmentation, pixel counts or patch-presence counts.
        alpha:
            Rebalancing strength.
            alpha=1.0 gives inverse frequency.
            alpha=0.5 gives sqrt inverse frequency.
            alpha=0.0 gives uniform over enabled classes.
        target_classes:
            Optional subset of classes to sample/rebalance.
            Example for binary foreground-only sampling: target_classes=[1].
        normalize:
            If True, return probabilities summing to 1.

    Returns:
        class_prob: [C]
    """
    counts = class_counts.detach().float().reshape(-1)
    num_classes = counts.numel()

    mask = _make_class_mask(
        num_classes,
        target_classes,
        device=counts.device,
        dtype=counts.dtype,
    )

    counts = counts.clamp_min(float(eps))
    weights = counts.pow(-float(alpha))
    weights = weights * mask

    if weights.sum() <= eps:
        if mask.sum() > 0:
            weights = mask / mask.sum().clamp_min(eps)
        else:
            weights = torch.ones_like(weights) / float(num_classes)

    if normalize:
        weights = weights / weights.sum().clamp_min(eps)

    return weights


def labels_to_class_counts(
    labels: Tensor,
    num_classes: int,
    *,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """Count image-level labels [N]."""
    labels = labels.long().reshape(-1)
    valid = (labels >= 0) & (labels < num_classes)
    if ignore_index is not None:
        valid = valid & (labels != int(ignore_index))
    labels = labels[valid]

    if labels.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32, device=labels.device)

    return torch.bincount(labels, minlength=num_classes).float()


def desired_class_prob_to_sample_weights(
    labels: Tensor,
    desired_class_prob: Tensor,
    *,
    class_counts: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    invalid_weight: float = 0.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Convert desired class probabilities into per-sample weights.

    Important:
        For WeightedRandomSampler, sample weight should be:

            w_i = P_desired(y_i) / n_{y_i}

        not simply P_desired(y_i). Otherwise majority classes still dominate
        because they have more samples.

    Args:
        labels:
            [N] image-level labels.
        desired_class_prob:
            [C] desired aggregate sampling probability per class.
        class_counts:
            Optional [C] counts. If None, computed from labels.
        ignore_index:
            Optional ignored label.
        invalid_weight:
            Weight assigned to invalid labels.
    """
    labels = labels.long().reshape(-1)
    desired = desired_class_prob.to(device=labels.device).float().reshape(-1)
    num_classes = desired.numel()

    if class_counts is None:
        class_counts = labels_to_class_counts(labels, num_classes, ignore_index=ignore_index)
    else:
        class_counts = class_counts.to(device=labels.device).float().reshape(-1)

    if class_counts.numel() != num_classes:
        raise ValueError("class_counts and desired_class_prob must have the same length")

    valid = (labels >= 0) & (labels < num_classes)
    if ignore_index is not None:
        valid = valid & (labels != int(ignore_index))

    weights = torch.full(
        labels.shape,
        float(invalid_weight),
        device=labels.device,
        dtype=desired.dtype,
    )

    per_class_sample_weight = desired / class_counts.clamp_min(eps)
    weights[valid] = per_class_sample_weight[labels[valid]]
    return weights


def labels_to_inverse_frequency_sample_weights(
    labels: Tensor,
    num_classes: int,
    *,
    alpha: float = 1.0,
    eps: float = 1e-6,
    target_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
    invalid_weight: float = 0.0,
) -> Tensor:
    """
    Full helper for image-level classification.

    It computes:
        class_counts
        desired_class_prob ∝ 1 / count_c^alpha
        sample_weight_i = desired_class_prob[y_i] / count[y_i]
    """
    class_counts = labels_to_class_counts(labels, num_classes, ignore_index=ignore_index)
    desired_class_prob = counts_to_inverse_frequency_prob(
        class_counts,
        alpha=alpha,
        eps=eps,
        target_classes=target_classes,
    )
    return desired_class_prob_to_sample_weights(
        labels,
        desired_class_prob,
        class_counts=class_counts,
        ignore_index=ignore_index,
        invalid_weight=invalid_weight,
        eps=eps,
    )


def segmentation_mask_to_frequency_weight(
    mask: Tensor,
    class_weight: Tensor,
    *,
    ignore_index: Optional[int] = 255,
    mode: Literal["max", "mean", "sum"] = "max",
    eps: float = 1e-6,
) -> Tensor:
    """
    Convert one segmentation mask/patch [H,W] to a scalar sampling weight.

    Args:
        mask:
            [H,W] mask.
        class_weight:
            [C] class weights, usually inverse-frequency probabilities.
        mode:
            "max":
                Use the largest class weight among classes present.
                Recommended for small-object patch sampling.
            "mean":
                Use pixel-histogram-weighted average.
            "sum":
                Sum weights of classes present.
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be [H,W], got {tuple(mask.shape)}")

    class_weight = class_weight.float()
    mask = mask.to(device=class_weight.device).long()
    num_classes = class_weight.numel()

    valid = (mask >= 0) & (mask < num_classes)
    if ignore_index is not None:
        valid = valid & (mask != int(ignore_index))

    labels = mask[valid]
    if labels.numel() == 0:
        return torch.tensor(float(eps), device=class_weight.device, dtype=class_weight.dtype)

    present = torch.unique(labels)

    if mode == "max":
        return class_weight[present].max().clamp_min(eps)

    if mode == "sum":
        return class_weight[present].sum().clamp_min(eps)

    if mode == "mean":
        hist = torch.bincount(labels, minlength=num_classes).to(
            device=class_weight.device,
            dtype=class_weight.dtype,
        )
        hist = hist / hist.sum().clamp_min(eps)
        return (hist * class_weight).sum().clamp_min(eps)

    raise ValueError("mode must be 'max', 'mean', or 'sum'")


def build_weighted_random_sampler(
    sample_weights: Tensor,
    *,
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build PyTorch WeightedRandomSampler from per-sample weights."""
    weights = sample_weights.detach().double().cpu()
    if num_samples is None:
        num_samples = int(weights.numel())
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples),
        replacement=bool(replacement),
    )