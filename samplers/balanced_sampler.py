"""Balanced sampling helpers for CCEL-Net baselines.

This file implements class-balanced sampling as a baseline/ablation.

Difference from other samplers:
    class_frequency_sampler.py:
        P(c) proportional to inverse class frequency.

    efficacy_guided_sampler.py:
        P(c) proportional to effective learning gap, e.g. (1 - psi_c)^alpha.

    balanced_sampler.py:
        P(c) = uniform over enabled classes.

For classification, we convert class-balanced probabilities into per-sample
weights. For segmentation, we provide patch-level balancing based on class
presence.
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence

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


def balanced_class_prob(
    num_classes: int,
    *,
    target_classes: Optional[Iterable[int]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
) -> Tensor:
    """
    Return uniform class probabilities over enabled classes.

    Example:
        num_classes=2, target_classes=None
            [0.5, 0.5]

        num_classes=2, target_classes=[1]
            [0.0, 1.0]
    """
    mask = _make_class_mask(
        num_classes,
        target_classes,
        device=device,
        dtype=dtype,
    )
    if mask.sum() <= eps:
        raise ValueError("target_classes produced an empty class mask")
    return mask / mask.sum().clamp_min(eps)


def labels_to_class_counts(
    labels: Tensor,
    num_classes: int,
    *,
    ignore_index: Optional[int] = None,
) -> Tensor:
    labels = labels.long().reshape(-1)
    valid = (labels >= 0) & (labels < num_classes)
    if ignore_index is not None:
        valid = valid & (labels != int(ignore_index))

    labels = labels[valid]
    if labels.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32, device=labels.device)

    return torch.bincount(labels, minlength=num_classes).float()


def labels_to_balanced_sample_weights(
    labels: Tensor,
    num_classes: int,
    *,
    target_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
    invalid_weight: float = 0.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Convert image labels [N] into per-sample weights for balanced class sampling.

    For WeightedRandomSampler, the correct per-sample weight is:

        w_i = P_balanced(y_i) / n_{y_i}

    This makes the aggregate probability of drawing each enabled class uniform.
    """
    labels = labels.long().reshape(-1)
    device = labels.device

    class_prob = balanced_class_prob(
        num_classes,
        target_classes=target_classes,
        device=device,
        dtype=torch.float32,
        eps=eps,
    )
    class_counts = labels_to_class_counts(labels, num_classes, ignore_index=ignore_index).to(device=device)

    valid = (labels >= 0) & (labels < num_classes)
    if ignore_index is not None:
        valid = valid & (labels != int(ignore_index))

    weights = torch.full(
        labels.shape,
        float(invalid_weight),
        device=device,
        dtype=torch.float32,
    )

    per_class_sample_weight = class_prob / class_counts.clamp_min(eps)
    weights[valid] = per_class_sample_weight[labels[valid]]

    # Samples from disabled classes receive zero weight.
    class_mask = (class_prob > 0).float()
    weights[valid] = weights[valid] * class_mask[labels[valid]]

    return weights


def mask_present_classes(
    mask: Tensor,
    num_classes: int,
    *,
    ignore_index: Optional[int] = 255,
) -> Tensor:
    """
    Return a [C] binary vector indicating which classes appear in a mask/patch.
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be [H,W], got {tuple(mask.shape)}")

    device = mask.device
    mask = mask.long()

    valid = (mask >= 0) & (mask < num_classes)
    if ignore_index is not None:
        valid = valid & (mask != int(ignore_index))

    labels = mask[valid]
    present = torch.zeros(num_classes, device=device, dtype=torch.float32)

    if labels.numel() == 0:
        return present

    present[torch.unique(labels)] = 1.0
    return present


def masks_to_class_presence_counts(
    masks: Sequence[Tensor],
    num_classes: int,
    *,
    ignore_index: Optional[int] = 255,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Count how many masks/patches contain each class.

    This is useful for segmentation patch-level balanced sampling.
    """
    if device is None:
        device = masks[0].device if len(masks) > 0 else torch.device("cpu")

    counts = torch.zeros(num_classes, device=device, dtype=torch.float32)

    for mask in masks:
        present = mask_present_classes(
            mask.to(device),
            num_classes,
            ignore_index=ignore_index,
        )
        counts += present

    return counts


def segmentation_mask_to_balanced_weight(
    mask: Tensor,
    class_presence_counts: Tensor,
    *,
    target_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = 255,
    mode: Literal["max", "mean", "sum"] = "max",
    eps: float = 1e-6,
) -> Tensor:
    """
    Convert one segmentation mask/patch to a scalar balanced sampling weight.

    This uses class-presence frequency across patches, not pixel frequency.

    For small-object segmentation, mode="max" is recommended:
        if a patch contains a rare class, the patch receives a high weight.

    Args:
        mask:
            [H,W] label mask.
        class_presence_counts:
            [C], number of patches containing each class.
        mode:
            "max": max inverse presence weight among present classes.
            "mean": average inverse presence weight among present classes.
            "sum": sum inverse presence weights among present classes.
    """
    class_presence_counts = class_presence_counts.float()
    num_classes = int(class_presence_counts.numel())
    device = class_presence_counts.device

    present = mask_present_classes(
        mask.to(device),
        num_classes,
        ignore_index=ignore_index,
    )

    class_mask = _make_class_mask(
        num_classes,
        target_classes,
        device=device,
        dtype=present.dtype,
    )

    present = present * class_mask
    if present.sum() <= eps:
        return torch.tensor(float(eps), device=device, dtype=torch.float32)

    inv_presence = 1.0 / class_presence_counts.clamp_min(eps)
    inv_presence = inv_presence * class_mask

    selected = inv_presence[present > 0]

    if selected.numel() == 0:
        return torch.tensor(float(eps), device=device, dtype=torch.float32)

    if mode == "max":
        return selected.max().clamp_min(eps)

    if mode == "mean":
        return selected.mean().clamp_min(eps)

    if mode == "sum":
        return selected.sum().clamp_min(eps)

    raise ValueError("mode must be 'max', 'mean', or 'sum'")


def masks_to_balanced_patch_weights(
    masks: Sequence[Tensor],
    num_classes: int,
    *,
    target_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = 255,
    mode: Literal["max", "mean", "sum"] = "max",
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute patch-level balanced sampling weights for a list of segmentation masks.
    """
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32)

    device = masks[0].device
    class_presence_counts = masks_to_class_presence_counts(
        masks,
        num_classes,
        ignore_index=ignore_index,
        device=device,
    )

    weights = []
    for mask in masks:
        w = segmentation_mask_to_balanced_weight(
            mask,
            class_presence_counts,
            target_classes=target_classes,
            ignore_index=ignore_index,
            mode=mode,
            eps=eps,
        )
        weights.append(w)

    return torch.stack(weights).float()


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