from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorLike = Union[Sequence[float], torch.Tensor]


def _squeeze_target(target: torch.Tensor) -> torch.Tensor:
    """
    Convert target from [B, 1, H, W] to [B, H, W] when needed.
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    return target


def _valid_mask(target: torch.Tensor, ignore_index: Optional[int]) -> torch.Tensor:
    target = _squeeze_target(target)
    if ignore_index is None:
        return torch.ones_like(target, dtype=torch.bool)
    return target != ignore_index


def _one_hot_encoder(
    target: torch.Tensor,
    n_classes: int,
    ignore_index: Optional[int] = 255,
) -> torch.Tensor:
    """
    target: [B, H, W] or [B, 1, H, W]
    return: [B, C, H, W]

    Ignored pixels are encoded as all-zero vectors.
    """
    target = _squeeze_target(target).long()
    valid = _valid_mask(target, ignore_index)

    target_safe = target.clone()
    target_safe[~valid] = 0
    target_safe = target_safe.clamp(min=0, max=n_classes - 1)

    one_hot = F.one_hot(target_safe, num_classes=n_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float().contiguous()
    one_hot = one_hot * valid.unsqueeze(1).float()

    return one_hot


def _build_class_weights(
    n_classes: int,
    class_counts: Optional[TensorLike] = None,
    class_priors: Optional[TensorLike] = None,
    mode: str = "inverse",
    beta: float = 0.9999,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build class weights for re-weighted CE.

    mode:
        inverse:
            w_c = 1 / freq_c

        sqrt_inverse:
            w_c = 1 / sqrt(freq_c)

        median_freq:
            w_c = median(freq) / freq_c

        effective_num:
            w_c = (1 - beta) / (1 - beta ^ n_c)

    class_counts is preferred.
    class_priors can also be used.
    """
    if class_counts is None and class_priors is None:
        return torch.ones(n_classes, dtype=torch.float32)

    if class_counts is not None:
        freq = torch.as_tensor(class_counts, dtype=torch.float32)
    else:
        freq = torch.as_tensor(class_priors, dtype=torch.float32)

    if freq.numel() != n_classes:
        raise ValueError(f"Expected {n_classes} class values, got {freq.numel()}.")

    freq = freq.clamp_min(eps)
    mode = mode.lower()

    if mode in {"inverse", "inv"}:
        weights = 1.0 / freq

    elif mode in {"sqrt_inverse", "sqrt_inv", "sqrt"}:
        weights = 1.0 / torch.sqrt(freq)

    elif mode in {"median_freq", "median_frequency"}:
        nonzero = freq[freq > eps]
        median = torch.median(nonzero) if nonzero.numel() > 0 else torch.tensor(1.0)
        weights = median / freq

    elif mode in {"effective_num", "effective", "cb"}:
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), freq))

    elif mode in {"none", "uniform"}:
        weights = torch.ones_like(freq)

    else:
        raise ValueError(
            f"Unknown weight mode: {mode}. "
            "Choose from inverse, sqrt_inverse, median_freq, effective_num, none."
        )

    if normalize:
        weights = weights / weights.mean().clamp_min(eps)

    return weights.float()


##### Binary segmentation #####
class DiceLoss_Binary(nn.Module):
    def __init__(self, ignore_index: Optional[int] = 255):
        super(DiceLoss_Binary, self).__init__()
        self.ignore_index = ignore_index

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss

        return loss

    def forward(self, inputs, target, weight=None, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        if target.dim() == 3:
            target = target.unsqueeze(1)

        target = target.float()

        if self.ignore_index is not None:
            valid = target != self.ignore_index
            target = torch.where(valid, target, torch.zeros_like(target))
            inputs = inputs * valid.float()

        assert inputs.size() == target.size(), \
            "predict {} & target {} shape do not match".format(inputs.size(), target.size())

        loss = self._dice_loss(inputs[:, 0], target[:, 0])

        if weight is not None:
            loss = loss * weight

        return loss


##### Multi-class segmentation #####
class DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index: Optional[int] = 255):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        return _one_hot_encoder(input_tensor, self.n_classes, self.ignore_index)

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss

        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        valid = _valid_mask(target, self.ignore_index).unsqueeze(1).float()
        inputs = inputs * valid

        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1.0] * self.n_classes

        weight = torch.as_tensor(weight, dtype=inputs.dtype, device=inputs.device)

        assert inputs.size() == target.size(), \
            "predict {} & target {} shape do not match".format(inputs.size(), target.size())

        loss = 0.0

        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]

        return loss / self.n_classes


##### Binary segmentation #####
class TverskyLoss_Binary(nn.Module):
    def __init__(self, alpha=0.7, ignore_index: Optional[int] = 255):
        super(TverskyLoss_Binary, self).__init__()
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.ignore_index = ignore_index

    def _tversky_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        TP = torch.sum(score * target)
        FP = torch.sum((1 - target) * score)
        FN = torch.sum(target * (1 - score))

        loss = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return 1 - loss

    def forward(self, inputs, target, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        if target.dim() == 3:
            target = target.unsqueeze(1)

        target = target.float()

        if self.ignore_index is not None:
            valid = target != self.ignore_index
            target = torch.where(valid, target, torch.zeros_like(target))
            inputs = inputs * valid.float()

        assert inputs.size() == target.size(), \
            "predict {} & target {} shape do not match".format(inputs.size(), target.size())

        loss = self._tversky_loss(inputs[:, 0], target[:, 0])

        return loss


##### Multi-class segmentation #####
class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.7, ignore_index: Optional[int] = 255):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        return _one_hot_encoder(input_tensor, self.n_classes, self.ignore_index)

    def _tversky_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        TP = torch.sum(score * target)
        FP = torch.sum((1 - target) * score)
        FN = torch.sum(target * (1 - score))

        loss = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return 1 - loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        valid = _valid_mask(target, self.ignore_index).unsqueeze(1).float()
        inputs = inputs * valid

        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), \
            "predict {} & target {} shape do not match".format(inputs.size(), target.size())

        loss = 0.0

        for i in range(self.n_classes):
            loss += self._tversky_loss(inputs[:, i], target[:, i])

        return loss / self.n_classes


##### tvMF Dice loss #####
class tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes, kappa=None, ignore_index: Optional[int] = 255):
        super(tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.kappa = 1.0 if kappa is None else kappa
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        return _one_hot_encoder(input_tensor, self.n_classes, self.ignore_index)

    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()

        score = F.normalize(score, p=2, dim=(0, 1, 2))
        target = F.normalize(target, p=2, dim=(0, 1, 2))

        cosine = torch.sum(score * target)

        intersect = (1.0 + cosine).div(1.0 + (1.0 - cosine).mul(kappa)) - 1.0
        loss = (1.0 - intersect) ** 2.0

        return loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        valid = _valid_mask(target, self.ignore_index).unsqueeze(1).float()
        inputs = inputs * valid

        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), \
            "predict {} & target {} shape do not match".format(inputs.size(), target.size())

        loss = 0.0

        for i in range(self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], self.kappa)
            loss += tvmf_dice

        return loss / self.n_classes


##### Adaptive tvMF Dice loss #####
class Adaptive_tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index: Optional[int] = 255):
        super(Adaptive_tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        return _one_hot_encoder(input_tensor, self.n_classes, self.ignore_index)

    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()

        score = F.normalize(score, p=2, dim=(0, 1, 2))
        target = F.normalize(target, p=2, dim=(0, 1, 2))

        cosine = torch.sum(score * target)

        intersect = (1.0 + cosine).div(1.0 + (1.0 - cosine).mul(kappa)) - 1.0
        loss = (1.0 - intersect) ** 2.0

        return loss

    def forward(self, inputs, target, kappa=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        valid = _valid_mask(target, self.ignore_index).unsqueeze(1).float()
        inputs = inputs * valid

        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), \
            "predict {} & target {} shape do not match".format(inputs.size(), target.size())

        if kappa is None:
            kappa = torch.ones(self.n_classes, dtype=inputs.dtype, device=inputs.device)
        else:
            kappa = torch.as_tensor(kappa, dtype=inputs.dtype, device=inputs.device)

        loss = 0.0

        for i in range(self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], kappa[i])
            loss += tvmf_dice

        return loss / self.n_classes


##### Re-weighted Cross Entropy #####
class ReweightedCELoss(nn.Module):
    """
    Re-weighted CE for semantic segmentation.

    Recommended input:
        logits: [B, C, H, W]
        target: [B, H, W] or [B, 1, H, W]

    For C >= 2:
        weighted CrossEntropyLoss.

    For C == 1:
        weighted BCEWithLogitsLoss.
    """

    def __init__(
        self,
        n_classes: Optional[int] = None,
        class_counts: Optional[TensorLike] = None,
        class_priors: Optional[TensorLike] = None,
        class_weights: Optional[TensorLike] = None,
        mode: str = "inverse",
        beta: float = 0.9999,
        ignore_index: Optional[int] = 255,
        reduction: str = "mean",
        normalize: bool = True,
        eps: float = 1e-6,
    ):
        super(ReweightedCELoss, self).__init__()

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

        if class_weights is not None:
            weight = torch.as_tensor(class_weights, dtype=torch.float32)

        elif n_classes is not None:
            weight = _build_class_weights(
                n_classes=n_classes,
                class_counts=class_counts,
                class_priors=class_priors,
                mode=mode,
                beta=beta,
                normalize=normalize,
                eps=eps,
            )

        else:
            weight = None

        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.float())

    def _binary_bce_forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)

        target = target.float()

        if self.ignore_index is not None:
            valid = target != self.ignore_index
            target_safe = torch.where(valid, target, torch.zeros_like(target))
        else:
            valid = torch.ones_like(target, dtype=torch.bool)
            target_safe = target

        loss = F.binary_cross_entropy_with_logits(
            inputs,
            target_safe,
            reduction="none",
        )

        if self.weight is not None:
            if self.weight.numel() == 2:
                w_bg = self.weight[0].to(inputs.device, inputs.dtype)
                w_fg = self.weight[1].to(inputs.device, inputs.dtype)

                pixel_weight = torch.where(target_safe > 0.5, w_fg, w_bg)
                loss = loss * pixel_weight

            elif self.weight.numel() == 1:
                loss = loss * self.weight[0].to(inputs.device, inputs.dtype)

            else:
                raise ValueError("For C == 1 BCE, class_weights should have length 1 or 2.")

        loss = loss[valid]

        if loss.numel() == 0:
            return inputs.sum() * 0.0

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "none":
            return loss

        raise ValueError(f"Unsupported reduction: {self.reduction}")

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = inputs.size(1)

        if c == 1:
            return self._binary_bce_forward(inputs, target)

        target = _squeeze_target(target).long()

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=inputs.device, dtype=inputs.dtype)

        return F.cross_entropy(
            inputs,
            target,
            weight=weight,
            ignore_index=-100 if self.ignore_index is None else self.ignore_index,
            reduction=self.reduction,
        )


##### LDAM / Margin-based Cross Entropy #####
class LDAMLoss(nn.Module):
    """
    LDAM loss adapted to pixel-wise semantic segmentation.

    Core idea:
        For each valid pixel, subtract class-dependent margin m_y from
        the logit of the ground-truth class, then apply CE.

    Recommended input:
        logits: [B, C, H, W], C >= 2
        target: [B, H, W] or [B, 1, H, W]

    class_counts should be pixel-level class counts from the training split.
    class_priors can also be used because LDAM margins are normalized by max_m.
    """

    def __init__(
        self,
        n_classes: int,
        class_counts: Optional[TensorLike] = None,
        class_priors: Optional[TensorLike] = None,
        max_m: float = 0.5,
        s: float = 30.0,
        class_weights: Optional[TensorLike] = None,
        weight_mode: Optional[str] = None,
        beta: float = 0.9999,
        ignore_index: Optional[int] = 255,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super(LDAMLoss, self).__init__()

        self.n_classes = n_classes
        self.max_m = max_m
        self.s = s
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

        if class_counts is None and class_priors is None:
            freq = torch.ones(n_classes, dtype=torch.float32)

        elif class_counts is not None:
            freq = torch.as_tensor(class_counts, dtype=torch.float32)

        else:
            freq = torch.as_tensor(class_priors, dtype=torch.float32)

        if freq.numel() != n_classes:
            raise ValueError(f"Expected {n_classes} class values, got {freq.numel()}.")

        freq = freq.clamp_min(eps)

        margins = 1.0 / torch.sqrt(torch.sqrt(freq))
        margins = margins * (max_m / margins.max().clamp_min(eps))

        self.register_buffer("margins", margins.float())

        if class_weights is not None:
            weight = torch.as_tensor(class_weights, dtype=torch.float32)

        elif weight_mode is not None:
            weight = _build_class_weights(
                n_classes=n_classes,
                class_counts=class_counts,
                class_priors=class_priors,
                mode=weight_mode,
                beta=beta,
                normalize=True,
                eps=eps,
            )

        else:
            weight = None

        if weight is None:
            self.register_buffer("weight", None)
        else:
            if weight.numel() != n_classes:
                raise ValueError(f"Expected {n_classes} class weights, got {weight.numel()}.")
            self.register_buffer("weight", weight.float())

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) < 2:
            raise ValueError(
                "LDAMLoss requires logits with C >= 2. "
                "For C == 1, use ReweightedCELoss / BCE instead."
            )

        target = _squeeze_target(target).long()
        valid = _valid_mask(target, self.ignore_index)

        target_safe = target.clone()
        target_safe[~valid] = 0
        target_safe = target_safe.clamp(min=0, max=self.n_classes - 1)

        index = F.one_hot(target_safe, num_classes=self.n_classes)
        index = index.permute(0, 3, 1, 2).to(dtype=inputs.dtype, device=inputs.device)
        index = index * valid.unsqueeze(1).to(dtype=inputs.dtype, device=inputs.device)

        margins = self.margins.to(device=inputs.device, dtype=inputs.dtype).view(1, -1, 1, 1)

        margin_logits = inputs - index * margins
        margin_logits = self.s * margin_logits

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=inputs.device, dtype=inputs.dtype)

        return F.cross_entropy(
            margin_logits,
            target,
            weight=weight,
            ignore_index=-100 if self.ignore_index is None else self.ignore_index,
            reduction=self.reduction,
        )


# Common aliases, convenient for config strings.
ReWeightedCELoss = ReweightedCELoss
WeightedCELoss = ReweightedCELoss

LDAMMarginLoss = LDAMLoss
MarginCELoss = LDAMLoss


__all__ = [
    "DiceLoss_Binary",
    "DiceLoss",
    "TverskyLoss_Binary",
    "TverskyLoss",
    "tvMF_DiceLoss",
    "Adaptive_tvMF_DiceLoss",
    "ReweightedCELoss",
    "ReWeightedCELoss",
    "WeightedCELoss",
    "LDAMLoss",
    "LDAMMarginLoss",
    "MarginCELoss",
]

"""
使用时建议这样写：
# Re-weighted CE
criterion = ReweightedCELoss(
    n_classes=num_classes,
    class_priors=train_priors,
    mode="inverse",
    ignore_index=255,
)

# LDAM / margin CE
criterion = LDAMLoss(
    n_classes=num_classes,
    class_priors=train_priors,
    max_m=0.5,
    s=30.0,
    ignore_index=255,
)"""

# 注意一点：LDAMLoss 要求模型输出是 [B, C, H, W]，并且 C >= 2。如果你的二分类模型输出是 [B, 1, H, W]，那 LDAM 不适合直接用；
# 要么把模型改成二通道输出 [B, 2, H, W]，要么二分类时只用 ReweightedCELoss 的 BCE 分支。