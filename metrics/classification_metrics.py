from __future__ import annotations

from typing import Dict, Optional, Sequence, Union, List

import math
import torch


TensorLike = Union[torch.Tensor, Sequence[int], Sequence[float]]


def _as_1d_long(x: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor to 1D long tensor.
    """
    if x.dim() > 1 and x.size(-1) == 1:
        x = x.squeeze(-1)

    return x.reshape(-1).long()


def _nanmean(x: torch.Tensor) -> torch.Tensor:
    """
    Mean while ignoring NaN values.
    If all values are NaN, return 0.
    """
    valid = ~torch.isnan(x)

    if valid.sum() == 0:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)

    return x[valid].mean()


def _safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-6,
    nan_if_zero: bool = True,
) -> torch.Tensor:
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


def hard_class_prediction(pred: torch.Tensor) -> torch.Tensor:
    """
    Convert classification output to hard labels.

    Supported:
        pred: [N, C]
            logits or probabilities, use argmax.

        pred: [N]
            already hard labels.

        pred: [N, 1]
            binary logits/probabilities or hard labels.
            If float:
                values outside [0, 1] are treated as logits and thresholded at 0.
                values inside [0, 1] are treated as probabilities and thresholded at 0.5.
    """
    if pred.dim() == 2:
        if pred.size(1) == 1:
            pred = pred.squeeze(1)

            if torch.is_floating_point(pred):
                with torch.no_grad():
                    is_logits = bool(pred.min() < 0 or pred.max() > 1)

                if is_logits:
                    return (pred > 0).long()

                return (pred > 0.5).long()

            return pred.long()

        return torch.argmax(pred, dim=1).long()

    if pred.dim() == 1:
        if torch.is_floating_point(pred):
            with torch.no_grad():
                is_binary_score = bool(pred.min() < 0 or pred.max() <= 1)

            if is_binary_score:
                if pred.min() < 0 or pred.max() > 1:
                    return (pred > 0).long()
                return (pred > 0.5).long()

        return pred.long()

    raise ValueError(
        f"Unsupported pred shape {pred.shape}. "
        "Expected [N, C], [N, 1], or [N]."
    )


def update_classification_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    confusion_matrix: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Update hard classification confusion matrix.

    Args:
        pred:
            [N, C] logits/probabilities or [N] hard labels.

        target:
            [N] ground-truth labels.

        num_classes:
            Number of classes.

        confusion_matrix:
            Existing confusion matrix. If None, create a new one.

        ignore_index:
            Optional ignored label.

    Returns:
        confusion_matrix:
            Shape [C, C].
            Rows are target classes.
            Columns are predicted classes.
    """
    pred_label = hard_class_prediction(pred)
    target = _as_1d_long(target)

    if pred_label.numel() != target.numel():
        raise ValueError(
            f"pred and target length mismatch: "
            f"pred={pred_label.numel()}, target={target.numel()}"
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


def classification_confusion_stats(
    confusion_matrix: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Convert classification confusion matrix to TP / FP / FN / TN.

    Args:
        confusion_matrix:
            Shape [C, C].
            Rows are target classes.
            Columns are predicted classes.

    Returns:
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


def build_shot_groups(
    class_counts: TensorLike,
    many_shot_thr: int = 100,
    few_shot_thr: int = 20,
) -> Dict[str, List[int]]:
    """
    Build many-shot / medium-shot / few-shot groups from training class counts.

    Default long-tailed classification convention:
        many-shot:
            count >= 100

        medium-shot:
            20 <= count < 100

        few-shot:
            count < 20

    Args:
        class_counts:
            Training-set sample count for each class.

    Returns:
        {
            "many": [...],
            "medium": [...],
            "few": [...]
        }
    """
    counts = torch.as_tensor(class_counts).reshape(-1)

    many = []
    medium = []
    few = []

    for c, count in enumerate(counts.tolist()):
        if count >= many_shot_thr:
            many.append(c)
        elif count < few_shot_thr:
            few.append(c)
        else:
            medium.append(c)

    return {
        "many": many,
        "medium": medium,
        "few": few,
    }


def build_head_medium_tail_groups(
    class_counts: TensorLike,
    head_thr: int = 100,
    tail_thr: int = 20,
) -> Dict[str, List[int]]:
    """
    Build head / medium / tail class groups from training class counts.

    By default, this is equivalent to:
        head:
            count >= 100

        medium:
            20 <= count < 100

        tail:
            count < 20

    Returns:
        {
            "head": [...],
            "medium": [...],
            "tail": [...]
        }
    """
    counts = torch.as_tensor(class_counts).reshape(-1)

    head = []
    medium = []
    tail = []

    for c, count in enumerate(counts.tolist()):
        if count >= head_thr:
            head.append(c)
        elif count < tail_thr:
            tail.append(c)
        else:
            medium.append(c)

    return {
        "head": head,
        "medium": medium,
        "tail": tail,
    }


def compute_group_accuracy(
    per_class_accuracy: torch.Tensor,
    support: torch.Tensor,
    correct: torch.Tensor,
    groups: Dict[str, Sequence[int]],
    prefix: str = "",
    eps: float = 1e-6,
) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Compute macro and micro accuracy for class groups.

    Macro group accuracy:
        mean of per-class accuracy inside this group.

    Micro group accuracy:
        total correct samples in group / total samples in group.

    Args:
        per_class_accuracy:
            Per-class accuracy, same as per-class recall.

        support:
            Number of ground-truth samples for each class.

        correct:
            Correct predictions for each class, usually TP.

        groups:
            Example:
                {
                    "many": [0, 1, 2],
                    "medium": [3, 4],
                    "few": [5]
                }

        prefix:
            Prefix for output keys.
    """
    output = {}

    device = per_class_accuracy.device

    for group_name, class_ids in groups.items():
        key = f"{prefix}{group_name}"

        if class_ids is None or len(class_ids) == 0:
            output[f"{key}_acc"] = torch.tensor(
                float("nan"),
                dtype=per_class_accuracy.dtype,
                device=device,
            )
            output[f"{key}_micro_acc"] = torch.tensor(
                float("nan"),
                dtype=per_class_accuracy.dtype,
                device=device,
            )
            output[f"{key}_num_classes"] = 0
            output[f"{key}_support"] = torch.tensor(
                0.0,
                dtype=support.dtype,
                device=device,
            )
            continue

        index = torch.as_tensor(class_ids, dtype=torch.long, device=device)

        group_acc_values = per_class_accuracy[index]
        group_support = support[index].float()
        group_correct = correct[index].float()

        macro_acc = _nanmean(group_acc_values)

        micro_acc = _safe_divide(
            group_correct.sum(),
            group_support.sum(),
            eps=eps,
            nan_if_zero=True,
        )

        output[f"{key}_acc"] = macro_acc
        output[f"{key}_micro_acc"] = micro_acc
        output[f"{key}_num_classes"] = len(class_ids)
        output[f"{key}_support"] = group_support.sum()

    return output


def compute_classification_metrics(
    confusion_matrix: torch.Tensor,
    class_counts: Optional[TensorLike] = None,
    shot_groups: Optional[Dict[str, Sequence[int]]] = None,
    head_medium_tail_groups: Optional[Dict[str, Sequence[int]]] = None,
    many_shot_thr: int = 100,
    few_shot_thr: int = 20,
    head_thr: int = 100,
    tail_thr: int = 20,
    eps: float = 1e-6,
) -> Dict[str, Union[torch.Tensor, int, Dict[str, Sequence[int]]]]:
    """
    Compute standard classification metrics from hard confusion matrix.

    Included:
        - overall accuracy
        - balanced accuracy
        - macro-F1
        - per-class accuracy
        - per-class precision / recall / F1
        - many-shot / medium-shot / few-shot accuracy
        - head / medium / tail group statistics

    Not included:
        - segmentation IoU
        - calibration ECE
        - evidence efficacy

    Args:
        confusion_matrix:
            Shape [C, C].
            Rows are target classes.
            Columns are predicted classes.

        class_counts:
            Training-set class counts.
            Required if you want automatic many/medium/few or head/medium/tail grouping.

        shot_groups:
            Optional explicit shot groups.
            Example:
                {
                    "many": [0, 1, 2],
                    "medium": [3, 4],
                    "few": [5, 6]
                }

        head_medium_tail_groups:
            Optional explicit head/medium/tail groups.
            Example:
                {
                    "head": [0, 1],
                    "medium": [2, 3],
                    "tail": [4, 5]
                }

    Returns:
        metrics dictionary.
    """
    cm = confusion_matrix

    if cm.dim() != 2 or cm.size(0) != cm.size(1):
        raise ValueError(f"confusion_matrix should be square [C, C], got {cm.shape}")

    cm = cm.float()
    num_classes = cm.size(0)

    stats = classification_confusion_stats(cm)

    tp = stats["tp"].float()
    fp = stats["fp"].float()
    fn = stats["fn"].float()
    tn = stats["tn"].float()

    support = stats["support"].float()
    pred_count = stats["pred_count"].float()
    total = stats["total"].float()

    # Per-class metrics.
    per_class_accuracy = _safe_divide(
        tp,
        support,
        eps=eps,
        nan_if_zero=True,
    )

    per_class_recall = per_class_accuracy

    per_class_precision = _safe_divide(
        tp,
        tp + fp,
        eps=eps,
        nan_if_zero=True,
    )

    per_class_f1 = _safe_divide(
        2.0 * per_class_precision * per_class_recall,
        per_class_precision + per_class_recall,
        eps=eps,
        nan_if_zero=True,
    )

    # Global metrics.
    overall_accuracy = _safe_divide(
        tp.sum(),
        total,
        eps=eps,
        nan_if_zero=False,
    )

    balanced_accuracy = _nanmean(per_class_recall)
    macro_f1 = _nanmean(per_class_f1)
    macro_precision = _nanmean(per_class_precision)
    macro_recall = balanced_accuracy

    metrics: Dict[str, Union[torch.Tensor, int, Dict[str, Sequence[int]]]] = {
        # Raw stats.
        "confusion_matrix": confusion_matrix,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": support,
        "pred_count": pred_count,
        "total": total,

        # Required classification metrics.
        "overall_accuracy": overall_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_F1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,

        # Per-class metrics.
        "per_class_accuracy": per_class_accuracy,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_F1": per_class_f1,
    }

    # Many-shot / medium-shot / few-shot.
    if shot_groups is None and class_counts is not None:
        shot_groups = build_shot_groups(
            class_counts=class_counts,
            many_shot_thr=many_shot_thr,
            few_shot_thr=few_shot_thr,
        )

    if shot_groups is not None:
        shot_metrics = compute_group_accuracy(
            per_class_accuracy=per_class_accuracy,
            support=support,
            correct=tp,
            groups=shot_groups,
            prefix="",
            eps=eps,
        )

        metrics.update(
            {
                "shot_groups": shot_groups,
                "many_shot_acc": shot_metrics.get("many_acc"),
                "medium_shot_acc": shot_metrics.get("medium_acc"),
                "few_shot_acc": shot_metrics.get("few_acc"),
                "many_shot_micro_acc": shot_metrics.get("many_micro_acc"),
                "medium_shot_micro_acc": shot_metrics.get("medium_micro_acc"),
                "few_shot_micro_acc": shot_metrics.get("few_micro_acc"),
                "many_shot_num_classes": shot_metrics.get("many_num_classes"),
                "medium_shot_num_classes": shot_metrics.get("medium_num_classes"),
                "few_shot_num_classes": shot_metrics.get("few_num_classes"),
                "many_shot_support": shot_metrics.get("many_support"),
                "medium_shot_support": shot_metrics.get("medium_support"),
                "few_shot_support": shot_metrics.get("few_support"),
            }
        )

    # Head / medium / tail.
    if head_medium_tail_groups is None and class_counts is not None:
        head_medium_tail_groups = build_head_medium_tail_groups(
            class_counts=class_counts,
            head_thr=head_thr,
            tail_thr=tail_thr,
        )

    if head_medium_tail_groups is not None:
        hmt_metrics = compute_group_accuracy(
            per_class_accuracy=per_class_accuracy,
            support=support,
            correct=tp,
            groups=head_medium_tail_groups,
            prefix="",
            eps=eps,
        )

        metrics.update(
            {
                "head_medium_tail_groups": head_medium_tail_groups,
                "head_acc": hmt_metrics.get("head_acc"),
                "medium_acc": hmt_metrics.get("medium_acc"),
                "tail_acc": hmt_metrics.get("tail_acc"),
                "head_micro_acc": hmt_metrics.get("head_micro_acc"),
                "medium_micro_acc": hmt_metrics.get("medium_micro_acc"),
                "tail_micro_acc": hmt_metrics.get("tail_micro_acc"),
                "head_num_classes": hmt_metrics.get("head_num_classes"),
                "medium_num_classes": hmt_metrics.get("medium_num_classes"),
                "tail_num_classes": hmt_metrics.get("tail_num_classes"),
                "head_support": hmt_metrics.get("head_support"),
                "medium_support": hmt_metrics.get("medium_support"),
                "tail_support": hmt_metrics.get("tail_support"),
            }
        )

    return metrics


def metrics_to_python(
    metrics: Dict[str, Union[torch.Tensor, int, float, Dict]],
) -> Dict[str, Union[float, int, torch.Tensor, Dict]]:
    """
    Convert scalar tensors to python floats.
    Keep vector tensors as CPU tensors.
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


def flatten_classification_metrics(
    metrics: Dict[str, Union[torch.Tensor, int, float, Dict]],
    class_names: Optional[Sequence[str]] = None,
    prefix: str = "",
) -> Dict[str, Union[float, int]]:
    """
    Flatten metrics for CSV / JSON / wandb logging.

    Example keys:
        overall_accuracy
        balanced_accuracy
        macro_F1
        class_0_acc
        class_0_F1
        class_0_P
        class_0_R
        many_shot_acc
        medium_shot_acc
        few_shot_acc
        head_acc
        medium_acc
        tail_acc
    """
    output = {}

    scalar_keys = [
        "overall_accuracy",
        "balanced_accuracy",
        "macro_F1",
        "macro_precision",
        "macro_recall",

        "many_shot_acc",
        "medium_shot_acc",
        "few_shot_acc",
        "many_shot_micro_acc",
        "medium_shot_micro_acc",
        "few_shot_micro_acc",
        "many_shot_num_classes",
        "medium_shot_num_classes",
        "few_shot_num_classes",
        "many_shot_support",
        "medium_shot_support",
        "few_shot_support",

        "head_acc",
        "medium_acc",
        "tail_acc",
        "head_micro_acc",
        "medium_micro_acc",
        "tail_micro_acc",
        "head_num_classes",
        "medium_num_classes",
        "tail_num_classes",
        "head_support",
        "medium_support",
        "tail_support",
    ]

    for key in scalar_keys:
        if key not in metrics:
            continue

        value = metrics[key]

        if value is None:
            continue

        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())

        output[prefix + key] = value

    per_class_accuracy = metrics.get("per_class_accuracy", None)
    per_class_precision = metrics.get("per_class_precision", None)
    per_class_recall = metrics.get("per_class_recall", None)
    per_class_f1 = metrics.get("per_class_F1", None)

    if per_class_accuracy is None:
        return output

    num_classes = int(per_class_accuracy.numel())

    for c in range(num_classes):
        if class_names is not None:
            name = str(class_names[c])
        else:
            name = f"class_{c}"

        name = name.replace(" ", "_")

        output[f"{prefix}{name}_acc"] = float(
            per_class_accuracy[c].detach().cpu().item()
        )

        if per_class_f1 is not None:
            output[f"{prefix}{name}_F1"] = float(
                per_class_f1[c].detach().cpu().item()
            )

        if per_class_precision is not None:
            output[f"{prefix}{name}_P"] = float(
                per_class_precision[c].detach().cpu().item()
            )

        if per_class_recall is not None:
            output[f"{prefix}{name}_R"] = float(
                per_class_recall[c].detach().cpu().item()
            )

    return output


class ClassificationMetric:
    """
    Stateful hard classification metric accumulator.

    This class only computes classification metrics from hard predictions.

    It does not compute:
        - segmentation IoU
        - calibration ECE
        - evidence efficacy

    Usage:
        metric = ClassificationMetric(
            num_classes=10,
            class_counts=train_class_counts,
        )

        for image, target in loader:
            logits = model(image)
            metric.update(logits, target)

        results = metric.compute()
        log_dict = metric.compute_flatten()
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: Optional[TensorLike] = None,
        shot_groups: Optional[Dict[str, Sequence[int]]] = None,
        head_medium_tail_groups: Optional[Dict[str, Sequence[int]]] = None,
        many_shot_thr: int = 100,
        few_shot_thr: int = 20,
        head_thr: int = 100,
        tail_thr: int = 20,
        ignore_index: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.long,
    ):
        self.num_classes = num_classes
        self.class_counts = class_counts
        self.shot_groups = shot_groups
        self.head_medium_tail_groups = head_medium_tail_groups
        self.many_shot_thr = many_shot_thr
        self.few_shot_thr = few_shot_thr
        self.head_thr = head_thr
        self.tail_thr = tail_thr
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
    ) -> torch.Tensor:
        self.matrix = update_classification_confusion_matrix(
            pred=pred,
            target=target,
            num_classes=self.num_classes,
            confusion_matrix=self.matrix,
            ignore_index=self.ignore_index,
        )

        return self.matrix

    def confusion_matrix(self) -> torch.Tensor:
        return self.matrix.clone()

    def compute(self, eps: float = 1e-6) -> Dict[str, Union[torch.Tensor, int, Dict]]:
        return compute_classification_metrics(
            confusion_matrix=self.matrix,
            class_counts=self.class_counts,
            shot_groups=self.shot_groups,
            head_medium_tail_groups=self.head_medium_tail_groups,
            many_shot_thr=self.many_shot_thr,
            few_shot_thr=self.few_shot_thr,
            head_thr=self.head_thr,
            tail_thr=self.tail_thr,
            eps=eps,
        )

    def compute_python(self, eps: float = 1e-6) -> Dict[str, Union[float, int, torch.Tensor, Dict]]:
        return metrics_to_python(self.compute(eps=eps))

    def compute_flatten(
        self,
        class_names: Optional[Sequence[str]] = None,
        prefix: str = "",
        eps: float = 1e-6,
    ) -> Dict[str, Union[float, int]]:
        metrics = self.compute(eps=eps)

        return flatten_classification_metrics(
            metrics=metrics,
            class_names=class_names,
            prefix=prefix,
        )

    def to(self, device: torch.device) -> "ClassificationMetric":
        self.matrix = self.matrix.to(device)
        self.device = device
        return self


__all__ = [
    "hard_class_prediction",
    "update_classification_confusion_matrix",
    "classification_confusion_stats",
    "build_shot_groups",
    "build_head_medium_tail_groups",
    "compute_group_accuracy",
    "compute_classification_metrics",
    "metrics_to_python",
    "flatten_classification_metrics",
    "ClassificationMetric",
]



# 使用示例：

# from classification_metrics import ClassificationMetric

# metric = ClassificationMetric(
#     num_classes=10,
#     class_counts=train_class_counts,
#     many_shot_thr=100,
#     few_shot_thr=20,
# )

# model.eval()
# metric.reset()

# with torch.no_grad():
#     for images, targets in val_loader:
#         images = images.cuda()
#         targets = targets.cuda()

#         logits = model(images)

#         metric.update(logits, targets)

# results = metric.compute()
# log_dict = metric.compute_flatten(prefix="val_")

# print("Confusion Matrix:")
# print(metric.confusion_matrix())

# print("Overall Acc:", results["overall_accuracy"])
# print("Balanced Acc:", results["balanced_accuracy"])
# print("Macro F1:", results["macro_F1"])

# print("Many-shot Acc:", results.get("many_shot_acc"))
# print("Medium-shot Acc:", results.get("medium_shot_acc"))
# print("Few-shot Acc:", results.get("few_shot_acc"))

# print("Head Acc:", results.get("head_acc"))
# print("Medium Acc:", results.get("medium_acc"))
# print("Tail Acc:", results.get("tail_acc"))

# print(log_dict)