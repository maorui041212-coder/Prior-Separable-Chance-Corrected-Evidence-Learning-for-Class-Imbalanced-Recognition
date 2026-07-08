from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ccel.losses.logit_adjustment_loss import (
    BalancedSoftmaxLoss,
    LogitAdjustedCrossEntropyLoss,
)
from ccel.losses.classification_mice_loss import (
    build_classification_mice_loss,
    MICELossClassification,
    ClassWiseMICELossClassification,
    CEMICECombinedClsLoss,
    CEMICEAblationClsLoss,
)

TensorLike = Union[Sequence[float], torch.Tensor]


def build_class_weights(
    num_classes: int,
    class_counts: Optional[TensorLike] = None,
    mode: str = "inverse",
    beta: float = 0.9999,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    if class_counts is None:
        weights = torch.ones(num_classes, dtype=torch.float32)
    else:
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(eps)

        if counts.numel() != num_classes:
            raise ValueError(f"Expected {num_classes} class counts, got {counts.numel()}.")

        mode = mode.lower()

        if mode in {"inverse", "inv"}:
            weights = 1.0 / counts

        elif mode in {"sqrt_inverse", "sqrt_inv", "sqrt"}:
            weights = 1.0 / torch.sqrt(counts)

        elif mode in {"effective_num", "effective", "cb"}:
            weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))

        elif mode in {"none", "uniform"}:
            weights = torch.ones_like(counts)

        else:
            raise ValueError(
                f"Unknown weight mode: {mode}. "
                "Choose from inverse, sqrt_inverse, effective_num, none."
            )

    if normalize:
        weights = weights / weights.mean().clamp_min(eps)

    return weights.float()


class ReweightedClassificationCELoss(nn.Module):
    """
    Re-weighting baseline for classification.

    logits: [B, C]
    target: [B]
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: TensorLike,
        mode: str = "inverse",
        beta: float = 0.9999,
        normalize: bool = True,
    ):
        super().__init__()

        weight = build_class_weights(
            num_classes=num_classes,
            class_counts=class_counts,
            mode=mode,
            beta=beta,
            normalize=normalize,
        )

        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=self.weight.to(device=logits.device, dtype=logits.dtype),
        )


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for classification.

    logits: [B, C]
    target: [B]
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[TensorLike] = None,
        reduction: str = "mean",
    ):
        super().__init__()

        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = alpha.gather(0, targets)
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "none":
            return loss

        raise ValueError(f"Unsupported reduction: {self.reduction}")


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number.

    Usually used with CE:
        CB-CE

    logits: [B, C]
    target: [B]
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: TensorLike,
        beta: float = 0.9999,
        loss_type: str = "ce",
        gamma: float = 2.0,
    ):
        super().__init__()

        self.loss_type = loss_type.lower()
        self.gamma = gamma

        weight = build_class_weights(
            num_classes=num_classes,
            class_counts=class_counts,
            mode="effective_num",
            beta=beta,
            normalize=True,
        )

        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        weight = self.weight.to(device=logits.device, dtype=logits.dtype)

        if self.loss_type == "ce":
            return F.cross_entropy(logits, targets, weight=weight)

        if self.loss_type == "focal":
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()

            log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
            pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)

            alpha_t = weight.gather(0, targets)
            loss = -alpha_t * ((1.0 - pt) ** self.gamma) * log_pt
            return loss.mean()

        raise ValueError(f"Unknown loss_type: {self.loss_type}")


class LDAMLoss(nn.Module):
    """
    LDAM loss for classification.

    logits: [B, C]
    target: [B]
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: TensorLike,
        max_m: float = 0.5,
        s: float = 30.0,
        weight: Optional[TensorLike] = None,
        eps: float = 1e-12,
    ):
        super().__init__()

        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(eps)

        if counts.numel() != num_classes:
            raise ValueError(f"Expected {num_classes} class counts, got {counts.numel()}.")

        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (max_m / margins.max().clamp_min(eps))

        self.num_classes = num_classes
        self.s = float(s)

        self.register_buffer("margins", margins.float())

        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()

        index = F.one_hot(targets, num_classes=self.num_classes).to(
            device=logits.device,
            dtype=logits.dtype,
        )

        margins = self.margins.to(device=logits.device, dtype=logits.dtype)

        margins_for_samples = margins.gather(0, targets).view(-1, 1)

        logits_m = logits - index * margins_for_samples
        logits_m = self.s * logits_m

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)

        return F.cross_entropy(logits_m, targets, weight=weight)


def build_classification_loss(
    name: str,
    num_classes: int,
    class_counts: TensorLike,
    logit_adjust_tau: float = 1.0,
    **kwargs,
):
    name = name.lower()

    if name in {"ce", "cross_entropy"}:
        return nn.CrossEntropyLoss()

    if name in {"reweight", "re_weighting", "reweighted_ce"}:
        return ReweightedClassificationCELoss(
            num_classes=num_classes,
            class_counts=class_counts,
            mode="inverse",
        )

    if name in {"balanced_softmax", "bs", "balanced_softmax_loss"}:
        return BalancedSoftmaxLoss(
            num_classes=num_classes,
            class_counts=class_counts,
            ignore_index=None,
            max_abs_adjustment=None,
        )

    if name in {"logit_adjustment", "la", "logit_adjusted_ce"}:
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        class_prior = counts / counts.sum().clamp_min(1.0)

        return LogitAdjustedCrossEntropyLoss(
            num_classes=num_classes,
            class_prior=class_prior,
            tau=logit_adjust_tau,
            direction="add",
            ignore_index=None,
            max_abs_adjustment=None,
        )

    if name in {"focal", "focal_loss"}:
        return FocalLoss(gamma=2.0)

    if name in {"cb", "class_balanced", "class_balanced_loss", "cb_ce"}:
        return ClassBalancedLoss(
            num_classes=num_classes,
            class_counts=class_counts,
            beta=0.9999,
            loss_type="ce",
        )

    if name in {"ldam", "ldam_loss"}:
        return LDAMLoss(
            num_classes=num_classes,
            class_counts=class_counts,
            max_m=0.5,
            s=30.0,
        )
    
    if name in {
        "mice",
        "ce_mice",
        "map_mice",
        "class_mice",
        "map_class_mice",
        "ce_map_mice",
        "ce_class_mice",
        "ce_map_class_mice",
    }:
        return build_classification_mice_loss(
            name=name,
            num_classes=num_classes,
            class_counts=class_counts,
            lambda_mice=kwargs.get("lambda_mice", 0.1),
            lambda_map=kwargs.get("lambda_map", 0.1),
            lambda_cls=kwargs.get("lambda_cls", 0.1),
            mice_mode=kwargs.get("mice_mode", "fixed"),
            smooth=kwargs.get("mice_smooth", 1e-6),
            ignore_index=None,
            upper_clamp_for_loss=kwargs.get("mice_upper_clamp_for_loss", True),
        )    

    raise ValueError(f"Unknown classification loss: {name}")


__all__ = [
    "build_class_weights",
    "ReweightedClassificationCELoss",
    "FocalLoss",
    "ClassBalancedLoss",
    "LDAMLoss",
    "build_classification_loss",
    "BalancedSoftmaxLoss",
    "LogitAdjustedCrossEntropyLoss",
    "MICELossClassification",
    "ClassWiseMICELossClassification",
    "CEMICECombinedClsLoss",
    "CEMICEAblationClsLoss",
]