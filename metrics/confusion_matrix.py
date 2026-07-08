from __future__ import annotations

from typing import Dict, Optional

import torch


def _squeeze_target(target: torch.Tensor) -> torch.Tensor:
    """
    Convert target from [B, 1, H, W] to [B, H, W] when needed.
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    return target


def _is_binary_float_prediction(pred: torch.Tensor, num_classes: int) -> bool:
    """
    Judge whether pred should be thresholded as binary prediction.
    """
    return num_classes == 2 and torch.is_floating_point(pred)


def _binary_threshold_prediction(
    pred: torch.Tensor,
    threshold: float = 0.5,
    from_logits: Optional[bool] = None,
) -> torch.Tensor:
    """
    Convert binary probability/logit map to hard labels.

    If from_logits is None:
        - values outside [0, 1] are treated as logits, threshold = 0
        - values inside [0, 1] are treated as probabilities, threshold = threshold

    Examples:
        logits: prob = sigmoid(logit), hard pred = logit > 0
        probs : hard pred = prob > 0.5
    """
    if from_logits is None:
        with torch.no_grad():
            min_val = pred.min()
            max_val = pred.max()
            auto_from_logits = bool(min_val < 0 or max_val > 1)
    else:
        auto_from_logits = bool(from_logits)

    if auto_from_logits:
        return (pred > 0).long()

    return (pred > threshold).long()


def hard_prediction(
    pred: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    binary_threshold: float = 0.5,
    binary_from_logits: Optional[bool] = None,
) -> torch.Tensor:
    """
    Convert model output or label map to hard prediction labels.

    Supported pred shapes:
        1. Multi-class logits/probabilities:
            pred: [B, C, H, W]
            output: argmax(pred, dim=1), shape [B, H, W]

        2. Binary one-channel logits/probabilities:
            pred: [B, 1, H, W]
            output: thresholded labels, shape [B, H, W]

        3. Already-hard labels:
            pred: [B, H, W] or [B, 1, H, W]
            output: [B, H, W]

    Notes:
        This function only creates hard prediction.
        It does not compute soft confusion or evidence efficacy.
    """
    if target is not None:
        target = _squeeze_target(target)

    # Case 1: pred has channel dimension.
    if target is not None and pred.dim() == target.dim() + 1:
        if pred.size(1) == 1:
            pred = pred.squeeze(1)

            if torch.is_floating_point(pred):
                return _binary_threshold_prediction(
                    pred,
                    threshold=binary_threshold,
                    from_logits=binary_from_logits,
                )

            return pred.long()

        return torch.argmax(pred, dim=1).long()

    # Case 2: pred is [B, 1, H, W] but target is not given.
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)

        if torch.is_floating_point(pred):
            return _binary_threshold_prediction(
                pred,
                threshold=binary_threshold,
                from_logits=binary_from_logits,
            )

        return pred.long()

    # Case 3: pred is already label map or binary score map.
    if pred.dim() == 4 and pred.size(1) != 1:
        return torch.argmax(pred, dim=1).long()

    if num_classes is not None and _is_binary_float_prediction(pred, num_classes):
        return _binary_threshold_prediction(
            pred,
            threshold=binary_threshold,
            from_logits=binary_from_logits,
        )

    return pred.long()


def update_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    confusion_matrix: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = 255,
    binary_threshold: float = 0.5,
    binary_from_logits: Optional[bool] = None,
) -> torch.Tensor:
    """
    Update hard confusion matrix.

    Args:
        pred:
            Model output or hard prediction.

            Supported shapes:
                [B, C, H, W]  multi-class logits/probabilities
                [B, 1, H, W]  binary logits/probabilities
                [B, H, W]     hard labels or binary probability map

        target:
            Ground truth labels.

            Supported shapes:
                [B, H, W]
                [B, 1, H, W]

        num_classes:
            Number of valid classes, excluding ignore_index.

        confusion_matrix:
            Existing confusion matrix. If None, a new one is created.

        ignore_index:
            Label value to ignore, usually 255.

        binary_threshold:
            Threshold for binary probability maps.

        binary_from_logits:
            Whether binary score maps are logits.

            True:
                threshold at 0

            False:
                threshold at binary_threshold

            None:
                automatically infer:
                    values outside [0, 1] -> logits
                    values inside [0, 1] -> probabilities

    Returns:
        confusion_matrix:
            Tensor with shape [num_classes, num_classes].
            Rows are ground truth classes.
            Columns are predicted classes.
    """
    target = _squeeze_target(target).long()

    pred_label = hard_prediction(
        pred=pred,
        target=target,
        num_classes=num_classes,
        binary_threshold=binary_threshold,
        binary_from_logits=binary_from_logits,
    )

    if pred_label.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch after hard prediction: "
            f"pred={pred_label.shape}, target={target.shape}"
        )

    device = target.device

    if confusion_matrix is None:
        confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.long,
            device=device,
        )
    else:
        confusion_matrix = confusion_matrix.to(device)

    if ignore_index is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = target != ignore_index

    # Also remove invalid class ids.
    valid = (
        valid
        & (target >= 0)
        & (target < num_classes)
        & (pred_label >= 0)
        & (pred_label < num_classes)
    )

    if valid.sum() == 0:
        return confusion_matrix

    target_valid = target[valid]
    pred_valid = pred_label[valid]

    indices = target_valid * num_classes + pred_valid

    hist = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    confusion_matrix += hist.to(confusion_matrix.dtype)

    return confusion_matrix


def confusion_matrix_stats(
    confusion_matrix: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute TP / FP / FN / TN from hard confusion matrix.

    For multi-class segmentation:
        TP / FP / FN / TN are computed in one-vs-rest manner for each class.

    Args:
        confusion_matrix:
            Tensor with shape [C, C].
            Rows are target classes.
            Columns are predicted classes.

    Returns:
        Dictionary containing:
            tp, fp, fn, tn, support, pred_count, total
    """
    cm = confusion_matrix

    if cm.dim() != 2 or cm.size(0) != cm.size(1):
        raise ValueError(f"confusion_matrix should be square [C, C], got {cm.shape}")

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    total = cm.sum()
    tn = total - tp - fp - fn

    support = cm.sum(dim=1)
    pred_count = cm.sum(dim=0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": support,
        "pred_count": pred_count,
        "total": total,
    }


def binary_confusion_stats(
    confusion_matrix: torch.Tensor,
    positive_class: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Return binary TP / FP / FN / TN for the selected positive class.

    For standard binary matrix:

        rows = target
        cols = pred

        [[TN, FP],
         [FN, TP]]

    positive_class=1 gives normal foreground-positive statistics.

    This function also works for multi-class confusion matrix by treating
    positive_class as one-vs-rest.
    """
    stats = confusion_matrix_stats(confusion_matrix)

    return {
        "tp": stats["tp"][positive_class],
        "fp": stats["fp"][positive_class],
        "fn": stats["fn"][positive_class],
        "tn": stats["tn"][positive_class],
        "support": stats["support"][positive_class],
        "pred_count": stats["pred_count"][positive_class],
        "total": stats["total"],
    }


def confusion_matrix_metrics(
    confusion_matrix: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Optional hard metrics from confusion matrix.

    This is still hard-confusion based.
    It does not include evidence efficacy or chance-corrected efficacy.
    """
    stats = confusion_matrix_stats(confusion_matrix)

    tp = stats["tp"].float()
    fp = stats["fp"].float()
    fn = stats["fn"].float()
    tn = stats["tn"].float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "iou": iou,
        "f1": f1,
        "accuracy": acc,
        "miou": iou.mean(),
        "mf1": f1.mean(),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_specificity": specificity.mean(),
    }


class ConfusionMatrix:
    """
    Stateful hard confusion matrix accumulator.

    Usage:
        cm = ConfusionMatrix(num_classes=2, ignore_index=255)

        for images, masks in loader:
            logits = model(images)
            cm.update(logits, masks)

        matrix = cm.compute()
        stats = cm.stats()
        metrics = cm.metrics()

    Notes:
        This class only uses hard predictions.
        It should not be used for evidence soft confusion.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = 255,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.long,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.dtype = dtype

        self.matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=dtype,
            device=device,
        )

    def reset(self) -> None:
        self.matrix.zero_()

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        binary_threshold: float = 0.5,
        binary_from_logits: Optional[bool] = None,
    ) -> torch.Tensor:
        self.matrix = update_confusion_matrix(
            pred=pred,
            target=target,
            num_classes=self.num_classes,
            confusion_matrix=self.matrix,
            ignore_index=self.ignore_index,
            binary_threshold=binary_threshold,
            binary_from_logits=binary_from_logits,
        )

        return self.matrix

    def compute(self) -> torch.Tensor:
        return self.matrix.clone()

    def stats(self) -> Dict[str, torch.Tensor]:
        return confusion_matrix_stats(self.matrix)

    def binary_stats(self, positive_class: int = 1) -> Dict[str, torch.Tensor]:
        return binary_confusion_stats(
            self.matrix,
            positive_class=positive_class,
        )

    def metrics(self, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
        return confusion_matrix_metrics(self.matrix, eps=eps)

    def to(self, device: torch.device) -> "ConfusionMatrix":
        self.matrix = self.matrix.to(device)
        self.device = device
        return self


__all__ = [
    "hard_prediction",
    "update_confusion_matrix",
    "confusion_matrix_stats",
    "binary_confusion_stats",
    "confusion_matrix_metrics",
    "ConfusionMatrix",
]