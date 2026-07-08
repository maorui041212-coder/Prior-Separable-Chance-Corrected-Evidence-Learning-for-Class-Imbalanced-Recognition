"""
Classification MICELoss for long-tailed image classification.

This file provides classification versions of the original MICE-style losses.

Supported logits:
    classification logits: [B, C]

Supported targets:
    classification labels: [B]

Main losses:
    - MICELossClassification
    - ClassWiseMICELossClassification
    - CEMICECombinedClsLoss
    - CEMICEAblationClsLoss

Recommended baseline name:
    Original MICELoss = CE + Map-wise MICE

For CIFAR-LT:
    logits:  [B, C]
    targets: [B]
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorLike = Union[Sequence[float], Sequence[int], torch.Tensor]


def _normalize_priors(
    class_priors: Optional[TensorLike],
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    if class_priors is None:
        priors = torch.ones(num_classes, dtype=torch.float32) / float(num_classes)
    else:
        priors = torch.as_tensor(class_priors, dtype=torch.float32).reshape(-1)
        if priors.numel() != num_classes:
            raise ValueError(
                f"class_priors length={priors.numel()} != num_classes={num_classes}"
            )
        priors = priors.clamp_min(smooth)
        priors = priors / priors.sum().clamp_min(smooth)

    return priors.float()


def _counts_to_priors(
    class_counts: TensorLike,
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32).reshape(-1)

    if counts.numel() != num_classes:
        raise ValueError(
            f"class_counts length={counts.numel()} != num_classes={num_classes}"
        )

    counts = counts.clamp_min(smooth)
    priors = counts / counts.sum().clamp_min(smooth)

    return priors.float()


def _squeeze_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.dim() == 2 and targets.size(1) == 1:
        targets = targets.squeeze(1)

    return targets.reshape(-1).long()


def _valid_classification_mask(
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    if ignore_index is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = targets != int(ignore_index)

    valid = valid & (targets >= 0) & (targets < num_classes)

    return valid


class MICELossClassification(nn.Module):
    """
    Map-wise MICE Loss for classification.

    Original MICE idea:
        MICE = (A - A0) / (1 - A0)

    Classification soft version:
        A_soft = mean_i p(y_i | x_i)
        A0     = sum_c pi_c^2
        Loss   = 1 - MICE

    Args:
        num_classes:
            Number of classes.

        class_priors:
            Training-set class prior pi_c. For CIFAR-LT, compute from train_set.class_counts.

        mice_mode:
            "fixed": use training-set prior.
            "batch": use current mini-batch prior.

        upper_clamp_for_loss:
            Clamp MICE score upper bound to 1 - smooth.
            Do not lower clamp, otherwise extreme imbalance may lose gradients.
    """

    def __init__(
        self,
        num_classes: int,
        class_priors: Optional[TensorLike] = None,
        mice_mode: str = "fixed",
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
        upper_clamp_for_loss: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.mice_mode = str(mice_mode).lower()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.upper_clamp_for_loss = bool(upper_clamp_for_loss)

        if self.mice_mode not in {"fixed", "batch"}:
            raise ValueError(f"Unsupported mice_mode: {self.mice_mode}")

        priors = _normalize_priors(
            class_priors=class_priors,
            num_classes=self.num_classes,
            smooth=self.smooth,
        )

        self.register_buffer("class_priors", priors)
        self.register_buffer("oa0", torch.sum(priors * priors))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_score: bool = False,
    ):
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        if logits.dim() != 2:
            raise ValueError(f"classification logits should be [B, C], got {logits.shape}")

        if logits.size(1) != self.num_classes:
            raise ValueError(
                f"logits channel={logits.size(1)} != num_classes={self.num_classes}"
            )

        targets = _squeeze_targets(targets).to(device=logits.device)

        if logits.size(0) != targets.numel():
            raise ValueError(
                f"logits batch={logits.size(0)} != targets numel={targets.numel()}"
            )

        valid = _valid_classification_mask(
            targets=targets,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        if valid.sum().item() == 0:
            zero = logits.sum() * 0.0
            if return_score:
                return zero, zero.detach()
            return zero

        logits_valid = logits[valid]
        targets_valid = targets[valid].long()

        probs = F.softmax(logits_valid, dim=1)

        correct_probs = probs.gather(
            dim=1,
            index=targets_valid.view(-1, 1),
        ).squeeze(1)

        soft_acc = correct_probs.mean()

        if self.mice_mode == "batch":
            counts = torch.bincount(
                targets_valid,
                minlength=self.num_classes,
            ).float().to(device=logits.device)

            priors = counts / counts.sum().clamp_min(self.smooth)
            oa0 = torch.sum(priors * priors)
        else:
            oa0 = self.oa0.to(device=logits.device, dtype=logits.dtype)

        raw_score = (soft_acc - oa0) / (1.0 - oa0 + self.smooth)

        if self.upper_clamp_for_loss:
            upper = torch.tensor(
                1.0 - self.smooth,
                device=raw_score.device,
                dtype=raw_score.dtype,
            )
            score_for_loss = torch.minimum(raw_score, upper)
        else:
            score_for_loss = raw_score

        loss = 1.0 - score_for_loss

        if return_score:
            return loss, raw_score.detach()

        return loss


class ClassWiseMICELossClassification(nn.Module):
    """
    Class-wise MICE Loss for classification.

    For each class c:
        RA_soft_c = mean p(c | x_i), where y_i = c
        A0_c      = pi_c
        MICE_c    = (RA_soft_c - A0_c) / (1 - A0_c)
        Loss_c    = 1 - MICE_c

    Then average over valid classes in the current batch.

    This gives each present class equal contribution, so tail classes are not
    overwhelmed by head classes.
    """

    def __init__(
        self,
        num_classes: int,
        class_priors: Optional[TensorLike] = None,
        mice_mode: str = "fixed",
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
        target_classes: Optional[Sequence[int]] = None,
        upper_clamp_for_loss: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.mice_mode = str(mice_mode).lower()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.upper_clamp_for_loss = bool(upper_clamp_for_loss)

        if self.mice_mode not in {"fixed", "batch"}:
            raise ValueError(f"Unsupported mice_mode: {self.mice_mode}")

        if target_classes is None:
            self.target_classes = None
        else:
            target_classes = [int(c) for c in target_classes]
            for c in target_classes:
                if c < 0 or c >= self.num_classes:
                    raise ValueError(f"target class {c} outside [0, {self.num_classes - 1}]")
            self.target_classes = target_classes

        priors = _normalize_priors(
            class_priors=class_priors,
            num_classes=self.num_classes,
            smooth=self.smooth,
        )

        self.register_buffer("class_priors", priors)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_score: bool = False,
    ):
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        if logits.dim() != 2:
            raise ValueError(f"classification logits should be [B, C], got {logits.shape}")

        if logits.size(1) != self.num_classes:
            raise ValueError(
                f"logits channel={logits.size(1)} != num_classes={self.num_classes}"
            )

        targets = _squeeze_targets(targets).to(device=logits.device)

        if logits.size(0) != targets.numel():
            raise ValueError(
                f"logits batch={logits.size(0)} != targets numel={targets.numel()}"
            )

        valid = _valid_classification_mask(
            targets=targets,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        if valid.sum().item() == 0:
            zero = logits.sum() * 0.0
            if return_score:
                return zero, zero.detach()
            return zero

        logits_valid = logits[valid]
        targets_valid = targets[valid].long()

        probs = F.softmax(logits_valid, dim=1)

        if self.mice_mode == "batch":
            counts = torch.bincount(
                targets_valid,
                minlength=self.num_classes,
            ).float().to(device=logits.device)

            priors = counts / counts.sum().clamp_min(self.smooth)
        else:
            priors = self.class_priors.to(device=logits.device, dtype=logits.dtype)

        if self.target_classes is None:
            class_iter = range(self.num_classes)
        else:
            class_iter = self.target_classes

        class_scores = []

        for c in class_iter:
            cls_mask = targets_valid == int(c)

            if cls_mask.sum().item() == 0:
                continue

            ra_soft_c = probs[cls_mask, int(c)].mean()

            a0_c = priors[int(c)].clamp(
                min=self.smooth,
                max=1.0 - self.smooth,
            )

            score_c = (ra_soft_c - a0_c) / (1.0 - a0_c + self.smooth)

            if self.upper_clamp_for_loss:
                upper = torch.tensor(
                    1.0 - self.smooth,
                    device=score_c.device,
                    dtype=score_c.dtype,
                )
                score_c = torch.minimum(score_c, upper)

            class_scores.append(score_c)

        if len(class_scores) == 0:
            zero = logits.sum() * 0.0
            if return_score:
                return zero, zero.detach()
            return zero

        class_score = torch.stack(class_scores).mean()
        loss = 1.0 - class_score

        if return_score:
            return loss, class_score.detach()

        return loss


class CEMICECombinedClsLoss(nn.Module):
    """
    CE / MICE / CE+MICE unified classification loss.

    loss_name:
        - ce
        - mice
        - ce_mice

    Here MICE means map-wise classification MICE:
        A_soft = mean p(y_i | x_i)
        A0     = sum_c pi_c^2
        MICE   = (A_soft - A0) / (1 - A0)
        Loss   = 1 - MICE
    """

    def __init__(
        self,
        num_classes: int,
        loss_name: str = "ce_mice",
        class_priors: Optional[TensorLike] = None,
        lambda_mice: float = 0.1,
        smooth: float = 1e-6,
        mice_mode: str = "fixed",
        ignore_index: Optional[int] = None,
        upper_clamp_for_loss: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.loss_name = str(loss_name).lower()
        self.lambda_mice = float(lambda_mice)
        self.ignore_index = ignore_index

        if self.loss_name not in {"ce", "mice", "ce_mice"}:
            raise ValueError(f"Unsupported loss_name: {self.loss_name}")

        if ignore_index is None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=int(ignore_index))

        self.mice = MICELossClassification(
            num_classes=num_classes,
            class_priors=class_priors,
            mice_mode=mice_mode,
            smooth=smooth,
            ignore_index=ignore_index,
            upper_clamp_for_loss=upper_clamp_for_loss,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_details: bool = False,
    ):
        ce_loss = self.ce(logits, _squeeze_targets(targets).to(device=logits.device))
        mice_loss, mice_score = self.mice(
            logits,
            targets,
            return_score=True,
        )

        if self.loss_name == "ce":
            total_loss = ce_loss
        elif self.loss_name == "mice":
            total_loss = mice_loss
        elif self.loss_name == "ce_mice":
            total_loss = ce_loss + self.lambda_mice * mice_loss
        else:
            raise RuntimeError(f"Unexpected loss_name={self.loss_name}")

        if return_details:
            return {
                "loss": total_loss,
                "ce_loss": ce_loss.detach(),
                "mice_loss": mice_loss.detach(),
                "mice_score": mice_score.detach(),
                "lambda_mice": self.lambda_mice,
            }

        return total_loss


class CEMICEAblationClsLoss(nn.Module):
    """
    Classification CE / Map-wise MICE / Class-wise MICE ablation loss.

    loss_name supports:
        - ce
        - ce_map_mice
        - ce_class_mice
        - ce_map_class_mice

    Optional pure MICE:
        - map_mice
        - class_mice
        - map_class_mice

    Recommended main baseline:
        - ce_map_mice

    This corresponds to the original metric-to-loss MICELoss baseline.
    """

    def __init__(
        self,
        num_classes: int,
        loss_name: str = "ce_map_mice",
        class_priors: Optional[TensorLike] = None,
        lambda_map: float = 0.1,
        lambda_cls: float = 0.1,
        smooth: float = 1e-6,
        mice_mode: str = "fixed",
        ignore_index: Optional[int] = None,
        target_classes: Optional[Sequence[int]] = None,
        upper_clamp_for_loss: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.loss_name = str(loss_name).lower()
        self.lambda_map = float(lambda_map)
        self.lambda_cls = float(lambda_cls)
        self.smooth = float(smooth)
        self.mice_mode = str(mice_mode).lower()
        self.ignore_index = ignore_index
        self.upper_clamp_for_loss = bool(upper_clamp_for_loss)

        alias = {
            "ce_mice": "ce_map_mice",
            "mice": "map_mice",
            "map": "map_mice",
            "class": "class_mice",
            "ce_map": "ce_map_mice",
            "ce_class": "ce_class_mice",
            "ce_all": "ce_map_class_mice",
            "ce_map_cls_mice": "ce_map_class_mice",
            "ce_map_class": "ce_map_class_mice",
            "map_class": "map_class_mice",
        }

        self.loss_name = alias.get(self.loss_name, self.loss_name)

        valid_loss_names = {
            "ce",
            "ce_map_mice",
            "ce_class_mice",
            "ce_map_class_mice",
            "map_mice",
            "class_mice",
            "map_class_mice",
        }

        if self.loss_name not in valid_loss_names:
            raise ValueError(
                f"Unsupported loss_name={self.loss_name}. "
                f"Supported: {sorted(valid_loss_names)}"
            )

        self.use_ce = self.loss_name in {
            "ce",
            "ce_map_mice",
            "ce_class_mice",
            "ce_map_class_mice",
        }

        self.use_map_mice = self.loss_name in {
            "ce_map_mice",
            "ce_map_class_mice",
            "map_mice",
            "map_class_mice",
        }

        self.use_class_mice = self.loss_name in {
            "ce_class_mice",
            "ce_map_class_mice",
            "class_mice",
            "map_class_mice",
        }

        if ignore_index is None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=int(ignore_index))

        if self.use_map_mice:
            self.map_mice = MICELossClassification(
                num_classes=num_classes,
                class_priors=class_priors,
                mice_mode=mice_mode,
                smooth=smooth,
                ignore_index=ignore_index,
                upper_clamp_for_loss=upper_clamp_for_loss,
            )
        else:
            self.map_mice = None

        if self.use_class_mice:
            self.class_mice = ClassWiseMICELossClassification(
                num_classes=num_classes,
                class_priors=class_priors,
                mice_mode=mice_mode,
                smooth=smooth,
                ignore_index=ignore_index,
                target_classes=target_classes,
                upper_clamp_for_loss=upper_clamp_for_loss,
            )
        else:
            self.class_mice = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_details: bool = False,
    ):
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        if logits.dim() != 2:
            raise ValueError(f"classification logits should be [B, C], got {logits.shape}")

        targets = _squeeze_targets(targets).to(device=logits.device)

        zero = logits.sum() * 0.0

        if self.use_ce:
            ce_loss = self.ce(logits, targets.long())
        else:
            ce_loss = zero

        if self.use_map_mice:
            map_mice_loss, map_mice_score = self.map_mice(
                logits,
                targets,
                return_score=True,
            )
        else:
            map_mice_loss = zero
            map_mice_score = zero.detach()

        if self.use_class_mice:
            class_mice_loss, class_mice_score = self.class_mice(
                logits,
                targets,
                return_score=True,
            )
        else:
            class_mice_loss = zero
            class_mice_score = zero.detach()

        if self.loss_name == "ce":
            total_loss = ce_loss

        elif self.loss_name == "ce_map_mice":
            total_loss = ce_loss + self.lambda_map * map_mice_loss

        elif self.loss_name == "ce_class_mice":
            total_loss = ce_loss + self.lambda_cls * class_mice_loss

        elif self.loss_name == "ce_map_class_mice":
            total_loss = (
                ce_loss
                + self.lambda_map * map_mice_loss
                + self.lambda_cls * class_mice_loss
            )

        elif self.loss_name == "map_mice":
            total_loss = map_mice_loss

        elif self.loss_name == "class_mice":
            total_loss = class_mice_loss

        elif self.loss_name == "map_class_mice":
            total_loss = (
                self.lambda_map * map_mice_loss
                + self.lambda_cls * class_mice_loss
            )

        else:
            raise RuntimeError(f"Unexpected loss_name={self.loss_name}")

        if return_details:
            return {
                "loss": total_loss,
                "ce_loss": ce_loss.detach(),
                "map_mice_loss": map_mice_loss.detach(),
                "map_mice_score": map_mice_score.detach(),
                "class_mice_loss": class_mice_loss.detach(),
                "class_mice_score": class_mice_score.detach(),
                "mice_loss": (map_mice_loss + class_mice_loss).detach(),
                "mice_score": (
                    map_mice_score.detach()
                    if self.use_map_mice
                    else class_mice_score.detach()
                ),
                "lambda_map": self.lambda_map,
                "lambda_cls": self.lambda_cls,
            }

        return total_loss


def build_classification_mice_loss(
    name: str,
    num_classes: int,
    class_counts: Optional[TensorLike] = None,
    *,
    lambda_mice: float = 0.1,
    lambda_map: float = 0.1,
    lambda_cls: float = 0.1,
    mice_mode: str = "fixed",
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
    upper_clamp_for_loss: bool = True,
) -> nn.Module:
    """
    Build classification MICELoss from class counts.

    name can be:
        ce_mice / ce_map_mice / ce_class_mice / ce_map_class_mice
        mice / map_mice / class_mice / map_class_mice
    """
    name = str(name).lower()

    if class_counts is None:
        class_priors = None
    else:
        class_priors = _counts_to_priors(
            class_counts=class_counts,
            num_classes=num_classes,
            smooth=smooth,
        )

    if name in {"ce", "mice", "ce_mice"}:
        return CEMICECombinedClsLoss(
            num_classes=num_classes,
            loss_name=name,
            class_priors=class_priors,
            lambda_mice=lambda_mice,
            smooth=smooth,
            mice_mode=mice_mode,
            ignore_index=ignore_index,
            upper_clamp_for_loss=upper_clamp_for_loss,
        )

    return CEMICEAblationClsLoss(
        num_classes=num_classes,
        loss_name=name,
        class_priors=class_priors,
        lambda_map=lambda_map,
        lambda_cls=lambda_cls,
        smooth=smooth,
        mice_mode=mice_mode,
        ignore_index=ignore_index,
        upper_clamp_for_loss=upper_clamp_for_loss,
    )


__all__ = [
    "MICELossClassification",
    "ClassWiseMICELossClassification",
    "CEMICECombinedClsLoss",
    "CEMICEAblationClsLoss",
    "build_classification_mice_loss",
]