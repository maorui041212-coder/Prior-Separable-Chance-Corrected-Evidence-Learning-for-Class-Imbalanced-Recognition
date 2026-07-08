from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import math
import torch


try:
    from .confusion_matrix import ConfusionMatrix
except ImportError:
    from confusion_matrix import ConfusionMatrix


TensorLike = Union[torch.Tensor, Sequence[float]]


def _to_float(value):
    """
    Convert scalar tensor to python float.
    Keep non-scalar tensor unchanged.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu()
    return value


def _safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-6,
    nan_if_zero: bool = True,
) -> torch.Tensor:
    """
    Safe division.

    If nan_if_zero=True:
        denominator == 0 gives NaN.
        This is useful for macro metrics, where absent classes should be ignored.

    If nan_if_zero=False:
        denominator == 0 gives 0.
    """
    numerator = numerator.float()
    denominator = denominator.float()

    if nan_if_zero:
        out = numerator / denominator.clamp_min(eps)
        out = torch.where(
            denominator > 0,
            out,
            torch.full_like(out, float("nan")),
        )
        return out

    return numerator / denominator.clamp_min(eps)


def _nanmean(x: torch.Tensor) -> torch.Tensor:
    """
    torch.nanmean wrapper with all-NaN protection.
    """
    valid = ~torch.isnan(x)

    if valid.sum() == 0:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)

    return x[valid].mean()


def _class_mask(
    num_classes: int,
    device: torch.device,
    include_background: bool = True,
    background_index: int = 0,
) -> torch.Tensor:
    """
    Build class mask for macro metrics.
    """
    mask = torch.ones(num_classes, dtype=torch.bool, device=device)

    if not include_background and 0 <= background_index < num_classes:
        mask[background_index] = False

    return mask


def confusion_matrix_to_stats(
    confusion_matrix: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Convert confusion matrix to TP / FP / FN / TN.

    Args:
        confusion_matrix:
            Shape [C, C].
            Rows are target classes.
            Columns are predicted classes.

    Returns:
        stats:
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


def infer_minority_class(
    confusion_matrix: torch.Tensor,
    class_priors: Optional[TensorLike] = None,
    background_index: int = 0,
    exclude_background: bool = True,
) -> int:
    """
    Infer minority class.

    Priority:
        1. If class_priors is provided:
            choose class with smallest prior.

        2. Else:
            choose class with smallest GT support from confusion matrix.

    By default, background class 0 is excluded.
    """
    cm = confusion_matrix
    num_classes = cm.size(0)
    device = cm.device

    candidate_mask = torch.ones(num_classes, dtype=torch.bool, device=device)

    if exclude_background and 0 <= background_index < num_classes:
        candidate_mask[background_index] = False

    if class_priors is not None:
        values = torch.as_tensor(class_priors, dtype=torch.float32, device=device)

        if values.numel() != num_classes:
            raise ValueError(
                f"class_priors length should be {num_classes}, got {values.numel()}"
            )

    else:
        values = cm.sum(dim=1).float()

        # Avoid selecting classes that never appear in GT.
        candidate_mask = candidate_mask & (values > 0)

    if candidate_mask.sum() == 0:
        # Fallback: use all classes.
        candidate_mask = torch.ones(num_classes, dtype=torch.bool, device=device)

        if exclude_background and 0 <= background_index < num_classes and num_classes > 1:
            candidate_mask[background_index] = False

    masked_values = values.clone()
    masked_values[~candidate_mask] = float("inf")

    minority_class = int(torch.argmin(masked_values).detach().cpu().item())

    return minority_class


def compute_segmentation_metrics(
    confusion_matrix: torch.Tensor,
    class_priors: Optional[TensorLike] = None,
    minority_class: Optional[int] = None,
    include_background: bool = True,
    background_index: int = 0,
    exclude_background_for_minority: bool = True,
    eps: float = 1e-6,
) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Compute standard semantic segmentation metrics from hard confusion matrix.

    Included:
        - per-class IoU
        - mIoU
        - per-class Precision
        - per-class Recall
        - per-class F1 / Dice
        - mF1 / mean Dice
        - OA
        - Kappa
        - Balanced Accuracy
        - minority IoU
        - minority F1

    Not included:
        - chance-corrected efficacy
        - evidence soft confusion
        - calibration metrics

    Args:
        confusion_matrix:
            Shape [C, C].
            Rows are target classes.
            Columns are predicted classes.

        class_priors:
            Optional class prior values.
            If provided and minority_class is None, minority class will be inferred
            from the smallest class prior.

        minority_class:
            Optional manually specified minority class id.

        include_background:
            Whether class 0 should be included in macro metrics such as mIoU and mF1.

        background_index:
            Background class id. Default is 0.

        exclude_background_for_minority:
            Whether background should be excluded when automatically selecting minority class.

    Returns:
        metrics:
            Dictionary containing tensor metrics and scalar metrics.
    """
    cm = confusion_matrix

    if cm.dim() != 2 or cm.size(0) != cm.size(1):
        raise ValueError(f"confusion_matrix should be square [C, C], got {cm.shape}")

    cm = cm.float()
    num_classes = cm.size(0)
    device = cm.device

    stats = confusion_matrix_to_stats(cm)

    tp = stats["tp"].float()
    fp = stats["fp"].float()
    fn = stats["fn"].float()
    tn = stats["tn"].float()
    support = stats["support"].float()
    pred_count = stats["pred_count"].float()
    total = stats["total"].float()

    # Per-class metrics.
    iou = _safe_divide(tp, tp + fp + fn, eps=eps, nan_if_zero=True)

    precision = _safe_divide(tp, tp + fp, eps=eps, nan_if_zero=True)
    recall = _safe_divide(tp, tp + fn, eps=eps, nan_if_zero=True)

    f1 = _safe_divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        eps=eps,
        nan_if_zero=True,
    )

    dice = f1

    specificity = _safe_divide(tn, tn + fp, eps=eps, nan_if_zero=True)

    # Macro class mask.
    macro_mask = _class_mask(
        num_classes=num_classes,
        device=device,
        include_background=include_background,
        background_index=background_index,
    )

    macro_iou = iou.clone()
    macro_f1 = f1.clone()
    macro_precision = precision.clone()
    macro_recall = recall.clone()
    macro_specificity = specificity.clone()

    macro_iou[~macro_mask] = float("nan")
    macro_f1[~macro_mask] = float("nan")
    macro_precision[~macro_mask] = float("nan")
    macro_recall[~macro_mask] = float("nan")
    macro_specificity[~macro_mask] = float("nan")

    miou = _nanmean(macro_iou)
    mf1 = _nanmean(macro_f1)
    mdice = mf1

    macro_precision_value = _nanmean(macro_precision)
    macro_recall_value = _nanmean(macro_recall)
    macro_specificity_value = _nanmean(macro_specificity)

    # Overall Accuracy.
    oa = _safe_divide(
        tp.sum(),
        total,
        eps=eps,
        nan_if_zero=False,
    )

    # Balanced Accuracy = mean per-class recall.
    balanced_accuracy = macro_recall_value

    # Cohen's Kappa from hard confusion matrix.
    if total <= 0:
        kappa = torch.tensor(0.0, dtype=cm.dtype, device=device)
    else:
        po = tp.sum() / total.clamp_min(eps)
        pe = (support * pred_count).sum() / (total * total).clamp_min(eps)

        if torch.abs(1.0 - pe) < eps:
            kappa = torch.tensor(0.0, dtype=cm.dtype, device=device)
        else:
            kappa = (po - pe) / (1.0 - pe)

    # Minority class.
    if minority_class is None:
        minority_class = infer_minority_class(
            confusion_matrix=cm,
            class_priors=class_priors,
            background_index=background_index,
            exclude_background=exclude_background_for_minority,
        )

    if minority_class < 0 or minority_class >= num_classes:
        raise ValueError(
            f"minority_class should be in [0, {num_classes - 1}], got {minority_class}"
        )

    minority_iou = iou[minority_class]
    minority_f1 = f1[minority_class]
    minority_precision = precision[minority_class]
    minority_recall = recall[minority_class]

    metrics = {
        # Raw confusion stats.
        "confusion_matrix": confusion_matrix,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": support,
        "pred_count": pred_count,
        "total": total,

        # Per-class segmentation metrics.
        "per_class_iou": iou,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_dice": dice,
        "per_class_specificity": specificity,

        # Macro metrics.
        "mIoU": miou,
        "mF1": mf1,
        "mDice": mdice,
        "macro_precision": macro_precision_value,
        "macro_recall": macro_recall_value,
        "macro_specificity": macro_specificity_value,

        # Global metrics.
        "OA": oa,
        "Kappa": kappa,
        "Balanced_Accuracy": balanced_accuracy,

        # Minority metrics.
        "minority_class": minority_class,
        "minority_IoU": minority_iou,
        "minority_F1": minority_f1,
        "minority_Dice": minority_f1,
        "minority_Precision": minority_precision,
        "minority_Recall": minority_recall,
    }

    return metrics


def metrics_to_python(
    metrics: Dict[str, Union[torch.Tensor, int, float]],
) -> Dict[str, Union[float, int, torch.Tensor]]:
    """
    Convert scalar tensors in metrics dictionary to python floats.

    Vector tensors such as per_class_iou are kept as CPU tensors.
    """
    output = {}

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                v = float(value.detach().cpu().item())

                if math.isnan(v):
                    v = float("nan")

                output[key] = v
            else:
                output[key] = value.detach().cpu()
        else:
            output[key] = value

    return output


def flatten_segmentation_metrics(
    metrics: Dict[str, Union[torch.Tensor, int, float]],
    class_names: Optional[Sequence[str]] = None,
    prefix: str = "",
) -> Dict[str, Union[float, int]]:
    """
    Flatten metrics for logging to CSV / JSON / wandb.

    Example output keys:
        mIoU
        mF1
        OA
        Kappa
        Balanced_Accuracy
        minority_class
        minority_IoU
        minority_F1
        class_0_IoU
        class_0_F1
        class_0_P
        class_0_R

    Args:
        metrics:
            Output of compute_segmentation_metrics.

        class_names:
            Optional class names.
            If provided, keys become:
                background_IoU, traffic_sign_IoU, ...

        prefix:
            Optional prefix, for example:
                "val_"
                "test_"
    """
    output = {}

    scalar_keys = [
        "mIoU",
        "mF1",
        "mDice",
        "macro_precision",
        "macro_recall",
        "macro_specificity",
        "OA",
        "Kappa",
        "Balanced_Accuracy",
        "minority_class",
        "minority_IoU",
        "minority_F1",
        "minority_Dice",
        "minority_Precision",
        "minority_Recall",
    ]

    for key in scalar_keys:
        if key in metrics:
            value = metrics[key]

            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().item())

            output[prefix + key] = value

    per_class_iou = metrics.get("per_class_iou", None)
    per_class_precision = metrics.get("per_class_precision", None)
    per_class_recall = metrics.get("per_class_recall", None)
    per_class_f1 = metrics.get("per_class_f1", None)
    per_class_dice = metrics.get("per_class_dice", None)

    if per_class_iou is None:
        return output

    num_classes = int(per_class_iou.numel())

    for c in range(num_classes):
        if class_names is not None:
            name = str(class_names[c])
        else:
            name = f"class_{c}"

        name = name.replace(" ", "_")

        output[f"{prefix}{name}_IoU"] = float(per_class_iou[c].detach().cpu().item())

        if per_class_f1 is not None:
            output[f"{prefix}{name}_F1"] = float(per_class_f1[c].detach().cpu().item())

        if per_class_dice is not None:
            output[f"{prefix}{name}_Dice"] = float(per_class_dice[c].detach().cpu().item())

        if per_class_precision is not None:
            output[f"{prefix}{name}_P"] = float(per_class_precision[c].detach().cpu().item())

        if per_class_recall is not None:
            output[f"{prefix}{name}_R"] = float(per_class_recall[c].detach().cpu().item())

    return output


class SegmentationMetric:
    """
    Hard segmentation metric accumulator.

    This class combines:
        - ConfusionMatrix from confusion_matrix.py
        - segmentation metrics from this file

    It still only uses hard prediction.
    It does not compute evidence efficacy or calibration.

    Usage:
        metric = SegmentationMetric(num_classes=2, ignore_index=255)

        for image, target in loader:
            logits = model(image)
            metric.update(logits, target)

        results = metric.compute()
        log_dict = metric.compute_flatten()
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = 255,
        class_priors: Optional[TensorLike] = None,
        minority_class: Optional[int] = None,
        include_background: bool = True,
        background_index: int = 0,
        exclude_background_for_minority: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_priors = class_priors
        self.minority_class = minority_class
        self.include_background = include_background
        self.background_index = background_index
        self.exclude_background_for_minority = exclude_background_for_minority

        self.cm = ConfusionMatrix(
            num_classes=num_classes,
            ignore_index=ignore_index,
            device=device,
        )

    def reset(self) -> None:
        self.cm.reset()

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        binary_threshold: float = 0.5,
        binary_from_logits: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Update hard confusion matrix using logits/probabilities/hard labels.
        """
        return self.cm.update(
            pred=pred,
            target=target,
            binary_threshold=binary_threshold,
            binary_from_logits=binary_from_logits,
        )

    def confusion_matrix(self) -> torch.Tensor:
        return self.cm.compute()

    def compute(self, eps: float = 1e-6) -> Dict[str, Union[torch.Tensor, int]]:
        matrix = self.cm.compute()

        return compute_segmentation_metrics(
            confusion_matrix=matrix,
            class_priors=self.class_priors,
            minority_class=self.minority_class,
            include_background=self.include_background,
            background_index=self.background_index,
            exclude_background_for_minority=self.exclude_background_for_minority,
            eps=eps,
        )

    def compute_python(self, eps: float = 1e-6) -> Dict[str, Union[float, int, torch.Tensor]]:
        return metrics_to_python(self.compute(eps=eps))

    def compute_flatten(
        self,
        class_names: Optional[Sequence[str]] = None,
        prefix: str = "",
        eps: float = 1e-6,
    ) -> Dict[str, Union[float, int]]:
        metrics = self.compute(eps=eps)

        return flatten_segmentation_metrics(
            metrics=metrics,
            class_names=class_names,
            prefix=prefix,
        )

    def to(self, device: torch.device) -> "SegmentationMetric":
        self.cm.to(device)
        return self


__all__ = [
    "confusion_matrix_to_stats",
    "infer_minority_class",
    "compute_segmentation_metrics",
    "metrics_to_python",
    "flatten_segmentation_metrics",
    "SegmentationMetric",
]

# 验证阶段可以这样用：

# from segmentation_metrics import SegmentationMetric

# metric = SegmentationMetric(
#     num_classes=2,
#     ignore_index=255,
#     class_priors=[0.9945, 0.0055],
#     minority_class=1,
#     include_background=True,
# )

# model.eval()
# metric.reset()

# with torch.no_grad():
#     for image, target in val_loader:
#         image = image.cuda()
#         target = target.cuda()

#         logits = model(image)

#         metric.update(logits, target)

# results = metric.compute()
# log_dict = metric.compute_flatten(prefix="val_")

# print("Confusion Matrix:")
# print(metric.confusion_matrix())

# print("mIoU:", results["mIoU"])
# print("mF1:", results["mF1"])
# print("OA:", results["OA"])
# print("Kappa:", results["Kappa"])
# print("Balanced Accuracy:", results["Balanced_Accuracy"])
# print("Minority IoU:", results["minority_IoU"])
# print("Minority F1:", results["minority_F1"])

# print(log_dict)