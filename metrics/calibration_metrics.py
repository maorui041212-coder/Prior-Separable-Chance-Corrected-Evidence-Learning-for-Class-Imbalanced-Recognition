from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import math
import torch
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, Sequence[float], Sequence[int]]


def _squeeze_target(target: torch.Tensor) -> torch.Tensor:
    """
    Convert target from [B, 1, H, W] to [B, H, W] when needed.
    """
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    if target.dim() == 2 and target.size(1) == 1:
        target = target.squeeze(1)

    return target


def _is_prob_tensor(
    x: torch.Tensor,
    dim: Optional[int] = None,
    atol: float = 1e-3,
) -> bool:
    """
    Roughly judge whether x looks like probability.

    For binary one-channel scores:
        only check whether all values are in [0, 1].

    For multi-class scores:
        check values in [0, 1] and sum approximately 1 along class dim.
    """
    if x.numel() == 0:
        return False

    with torch.no_grad():
        min_val = float(x.min().detach().cpu().item())
        max_val = float(x.max().detach().cpu().item())

        if min_val < -atol or max_val > 1.0 + atol:
            return False

        if dim is None:
            return True

        prob_sum = x.sum(dim=dim)
        ones = torch.ones_like(prob_sum)

        return bool(torch.allclose(prob_sum, ones, atol=atol, rtol=atol))


def _binary_scores_to_probs(
    scores: torch.Tensor,
    input_type: str = "auto",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Convert binary scores to two-column probabilities [P(class 0), P(class 1)].

    scores:
        [N] or any flattened shape.

    input_type:
        "logits":
            scores are binary logits.

        "probs":
            scores are positive-class probabilities.

        "auto":
            values inside [0, 1] -> probabilities
            otherwise -> logits
    """
    scores = scores.float()

    if input_type not in {"auto", "logits", "probs"}:
        raise ValueError(f"input_type should be auto/logits/probs, got {input_type}")

    if input_type == "auto":
        is_probs = _is_prob_tensor(scores, dim=None)
    else:
        is_probs = input_type == "probs"

    if is_probs:
        p1 = scores.clamp(min=eps, max=1.0 - eps)
    else:
        p1 = torch.sigmoid(scores).clamp(min=eps, max=1.0 - eps)

    p0 = 1.0 - p1

    return torch.stack([p0, p1], dim=1)


def _multiclass_scores_to_probs(
    scores: torch.Tensor,
    input_type: str = "auto",
    class_dim: int = 1,
    eps: float = 1e-12,
    normalize_probs: bool = True,
) -> torch.Tensor:
    """
    Convert multi-class logits/probabilities to probabilities.
    """
    if input_type not in {"auto", "logits", "probs"}:
        raise ValueError(f"input_type should be auto/logits/probs, got {input_type}")

    scores = scores.float()

    if input_type == "auto":
        is_probs = _is_prob_tensor(scores, dim=class_dim)
    else:
        is_probs = input_type == "probs"

    if is_probs:
        probs = scores.clamp_min(eps)

        if normalize_probs:
            probs = probs / probs.sum(dim=class_dim, keepdim=True).clamp_min(eps)

        return probs.clamp(min=eps, max=1.0)

    return torch.softmax(scores, dim=class_dim).clamp(min=eps, max=1.0)


def flatten_probs_and_target(
    scores: torch.Tensor,
    target: torch.Tensor,
    input_type: str = "auto",
    ignore_index: Optional[int] = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert classification or segmentation outputs into flat probabilities and labels.

    Supported scores:
        Classification:
            [N, C]
            [N, 1]
            [N]

        Segmentation:
            [B, C, H, W]
            [B, 1, H, W]
            [B, H, W]

    Supported target:
        Classification:
            [N]
            [N, 1]

        Segmentation:
            [B, H, W]
            [B, 1, H, W]

    Returns:
        probs:
            [M, C]

        target:
            [M]
    """
    target = _squeeze_target(target).long()

    # Case 1: segmentation or dense prediction with explicit channel dim.
    # scores: [B, C, H, W], target: [B, H, W]
    if scores.dim() == target.dim() + 1:
        class_dim = 1

        if scores.size(class_dim) == 1:
            binary_scores = scores.squeeze(class_dim).reshape(-1)
            probs = _binary_scores_to_probs(
                binary_scores,
                input_type=input_type,
                eps=eps,
            )
        else:
            probs_dense = _multiclass_scores_to_probs(
                scores,
                input_type=input_type,
                class_dim=class_dim,
                eps=eps,
            )

            # [B, C, H, W] -> [B, H, W, C] -> [M, C]
            probs = probs_dense.permute(0, *range(2, probs_dense.dim()), 1)
            probs = probs.reshape(-1, scores.size(class_dim))

        target_flat = target.reshape(-1)

    # Case 2: classification [N, C] or [N, 1]
    elif scores.dim() == 2:
        if scores.size(1) == 1:
            probs = _binary_scores_to_probs(
                scores.squeeze(1),
                input_type=input_type,
                eps=eps,
            )
        else:
            probs = _multiclass_scores_to_probs(
                scores,
                input_type=input_type,
                class_dim=1,
                eps=eps,
            )

        target_flat = target.reshape(-1)

    # Case 3: binary scores or hard-ish probability maps without channel.
    elif scores.dim() == target.dim():
        probs = _binary_scores_to_probs(
            scores.reshape(-1),
            input_type=input_type,
            eps=eps,
        )

        target_flat = target.reshape(-1)

    else:
        raise ValueError(
            f"Unsupported scores/target shapes: scores={scores.shape}, target={target.shape}"
        )

    if probs.size(0) != target_flat.numel():
        raise ValueError(
            f"Flattened scores and target length mismatch: "
            f"probs={probs.size(0)}, target={target_flat.numel()}"
        )

    num_classes = probs.size(1)

    if ignore_index is None:
        valid = torch.ones_like(target_flat, dtype=torch.bool)
    else:
        valid = target_flat != ignore_index

    valid = valid & (target_flat >= 0) & (target_flat < num_classes)

    probs = probs[valid]
    target_flat = target_flat[valid]

    return probs, target_flat


def reliability_diagram_stats_from_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 15,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute reliability diagram bin statistics from probabilities.

    Args:
        probs:
            [N, C] probabilities.

        target:
            [N] ground-truth labels.

        n_bins:
            Number of confidence bins.

    Returns:
        Dictionary containing:
            bin_lower
            bin_upper
            bin_count
            bin_accuracy
            bin_confidence
            bin_gap
            bin_proportion
            confidence
            prediction
            correctness
    """
    if probs.dim() != 2:
        raise ValueError(f"probs should be [N, C], got {probs.shape}")

    target = target.reshape(-1).long()

    if probs.size(0) != target.numel():
        raise ValueError(
            f"probs and target length mismatch: probs={probs.size(0)}, target={target.numel()}"
        )

    device = probs.device

    if probs.size(0) == 0:
        bin_lower = torch.linspace(0.0, 1.0, n_bins + 1, device=device)[:-1]
        bin_upper = torch.linspace(0.0, 1.0, n_bins + 1, device=device)[1:]

        empty = torch.zeros(n_bins, dtype=torch.float32, device=device)

        return {
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "bin_count": empty.long(),
            "bin_accuracy": torch.full_like(empty, float("nan")),
            "bin_confidence": torch.full_like(empty, float("nan")),
            "bin_gap": torch.full_like(empty, float("nan")),
            "bin_proportion": empty,
            "confidence": torch.empty(0, dtype=torch.float32, device=device),
            "prediction": torch.empty(0, dtype=torch.long, device=device),
            "correctness": torch.empty(0, dtype=torch.float32, device=device),
        }

    confidence, prediction = torch.max(probs, dim=1)
    correctness = prediction.eq(target).float()

    confidence = confidence.clamp(min=0.0, max=1.0)

    # Equal-width bins over [0, 1].
    # confidence == 1.0 goes into the last bin.
    bin_index = torch.clamp((confidence * n_bins).long(), max=n_bins - 1)

    bin_count = torch.bincount(
        bin_index,
        minlength=n_bins,
    ).to(device=device)

    bin_conf_sum = torch.zeros(n_bins, dtype=torch.float32, device=device)
    bin_acc_sum = torch.zeros(n_bins, dtype=torch.float32, device=device)

    bin_conf_sum.scatter_add_(0, bin_index, confidence.float())
    bin_acc_sum.scatter_add_(0, bin_index, correctness.float())

    nonempty = bin_count > 0

    bin_confidence = torch.full(
        (n_bins,),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    bin_accuracy = torch.full(
        (n_bins,),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )

    bin_confidence[nonempty] = bin_conf_sum[nonempty] / bin_count[nonempty].float().clamp_min(eps)
    bin_accuracy[nonempty] = bin_acc_sum[nonempty] / bin_count[nonempty].float().clamp_min(eps)

    bin_gap = torch.abs(bin_accuracy - bin_confidence)

    bin_proportion = bin_count.float() / float(probs.size(0))

    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    bin_lower = boundaries[:-1]
    bin_upper = boundaries[1:]

    return {
        "bin_lower": bin_lower,
        "bin_upper": bin_upper,
        "bin_count": bin_count,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_gap": bin_gap,
        "bin_proportion": bin_proportion,
        "confidence": confidence.detach(),
        "prediction": prediction.detach(),
        "correctness": correctness.detach(),
    }


def calibration_error_from_bin_stats(
    bin_count: torch.Tensor,
    bin_accuracy: torch.Tensor,
    bin_confidence: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute ECE and MCE from reliability bin statistics.

    ECE:
        sum_b |acc(b) - conf(b)| * n_b / N

    MCE:
        max_b |acc(b) - conf(b)| over non-empty bins
    """
    device = bin_count.device
    total = bin_count.sum().float()

    if total <= 0:
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        return {
            "ECE": zero,
            "MCE": zero,
        }

    nonempty = bin_count > 0

    gap = torch.abs(bin_accuracy - bin_confidence)
    gap = torch.where(nonempty, gap, torch.zeros_like(gap))

    weight = bin_count.float() / total.clamp_min(eps)

    ece = torch.sum(weight * gap)

    if nonempty.sum() == 0:
        mce = torch.tensor(0.0, dtype=torch.float32, device=device)
    else:
        mce = gap[nonempty].max()

    return {
        "ECE": ece,
        "MCE": mce,
    }


def nll_from_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Negative log likelihood from probabilities.

    probs:
        [N, C]

    target:
        [N]
    """
    if probs.size(0) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=probs.device)

    target = target.reshape(-1).long()

    p_true = probs[torch.arange(probs.size(0), device=probs.device), target]
    nll = -torch.log(p_true.clamp_min(eps)).mean()

    return nll


def brier_score_from_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-class Brier score.

    Definition:
        mean_i sum_c (p_ic - 1[y_i = c])^2

    For binary classification, probs should still be [N, 2].
    """
    if probs.size(0) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=probs.device)

    target = target.reshape(-1).long()
    num_classes = probs.size(1)

    one_hot = F.one_hot(target, num_classes=num_classes).float()
    brier = torch.sum((probs - one_hot) ** 2, dim=1).mean()

    return brier


def compute_calibration_metrics_from_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 15,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute calibration metrics from probabilities.

    Included:
        - ECE
        - MCE
        - NLL
        - Brier score
        - reliability diagram statistics

    Not included:
        - mIoU / F1
        - evidence efficacy
    """
    probs = probs.float()
    target = target.reshape(-1).long()

    if probs.dim() != 2:
        raise ValueError(f"probs should be [N, C], got {probs.shape}")

    if probs.size(0) != target.numel():
        raise ValueError(
            f"probs and target length mismatch: probs={probs.size(0)}, target={target.numel()}"
        )

    reliability = reliability_diagram_stats_from_probs(
        probs=probs,
        target=target,
        n_bins=n_bins,
        eps=eps,
    )

    cal_error = calibration_error_from_bin_stats(
        bin_count=reliability["bin_count"],
        bin_accuracy=reliability["bin_accuracy"],
        bin_confidence=reliability["bin_confidence"],
        eps=eps,
    )

    nll = nll_from_probs(
        probs=probs,
        target=target,
        eps=eps,
    )

    brier = brier_score_from_probs(
        probs=probs,
        target=target,
    )

    metrics = {
        "ECE": cal_error["ECE"],
        "MCE": cal_error["MCE"],
        "NLL": nll,
        "Brier": brier,
        "num_samples": torch.tensor(
            probs.size(0),
            dtype=torch.long,
            device=probs.device,
        ),
        "n_bins": torch.tensor(
            n_bins,
            dtype=torch.long,
            device=probs.device,
        ),
    }

    metrics.update(reliability)

    return metrics


def compute_calibration_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 15,
    input_type: str = "auto",
    ignore_index: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute calibration metrics from logits/probabilities.

    Args:
        scores:
            Classification:
                [N, C], [N, 1], or [N]

            Segmentation:
                [B, C, H, W], [B, 1, H, W], or [B, H, W]

        target:
            Classification:
                [N] or [N, 1]

            Segmentation:
                [B, H, W] or [B, 1, H, W]

        n_bins:
            Number of bins for ECE/MCE/reliability diagram.

        input_type:
            "auto":
                automatically infer logits/probabilities.

            "logits":
                scores are logits.

            "probs":
                scores are probabilities.

        ignore_index:
            Optional ignored target label.
            For segmentation this is usually 255.

    Returns:
        Calibration metric dictionary.
    """
    probs, target_flat = flatten_probs_and_target(
        scores=scores,
        target=target,
        input_type=input_type,
        ignore_index=ignore_index,
        eps=eps,
    )

    return compute_calibration_metrics_from_probs(
        probs=probs,
        target=target_flat,
        n_bins=n_bins,
        eps=eps,
    )


def reliability_diagram_table(
    metrics: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Extract only reliability diagram table fields.

    Useful fields:
        bin_lower
        bin_upper
        bin_count
        bin_proportion
        bin_accuracy
        bin_confidence
        bin_gap
    """
    keys = [
        "bin_lower",
        "bin_upper",
        "bin_count",
        "bin_proportion",
        "bin_accuracy",
        "bin_confidence",
        "bin_gap",
    ]

    return {key: metrics[key] for key in keys if key in metrics}


def metrics_to_python(
    metrics: Dict[str, torch.Tensor],
) -> Dict[str, Union[float, int, torch.Tensor]]:
    """
    Convert scalar tensors to python numbers.
    Keep vector tensors as CPU tensors.
    """
    output = {}

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                if value.dtype in {torch.long, torch.int, torch.int64, torch.int32}:
                    output[key] = int(value.detach().cpu().item())
                else:
                    v = float(value.detach().cpu().item())

                    if math.isnan(v):
                        v = float("nan")

                    output[key] = v
            else:
                output[key] = value.detach().cpu()
        else:
            output[key] = value

    return output


def flatten_calibration_metrics(
    metrics: Dict[str, torch.Tensor],
    prefix: str = "",
    include_bins: bool = True,
) -> Dict[str, Union[float, int]]:
    """
    Flatten calibration metrics for CSV / JSON / wandb logging.

    Example keys:
        val_ECE
        val_MCE
        val_NLL
        val_Brier
        val_bin_00_count
        val_bin_00_acc
        val_bin_00_conf
        val_bin_00_gap
    """
    output: Dict[str, Union[float, int]] = {}

    scalar_keys = [
        "ECE",
        "MCE",
        "NLL",
        "Brier",
        "num_samples",
        "n_bins",
    ]

    for key in scalar_keys:
        if key not in metrics:
            continue

        value = metrics[key]

        if isinstance(value, torch.Tensor):
            if value.dtype in {torch.long, torch.int, torch.int64, torch.int32}:
                value = int(value.detach().cpu().item())
            else:
                value = float(value.detach().cpu().item())

        output[prefix + key] = value

    if not include_bins:
        return output

    required = [
        "bin_lower",
        "bin_upper",
        "bin_count",
        "bin_proportion",
        "bin_accuracy",
        "bin_confidence",
        "bin_gap",
    ]

    if not all(key in metrics for key in required):
        return output

    n_bins = int(metrics["bin_count"].numel())

    for i in range(n_bins):
        output[f"{prefix}bin_{i:02d}_lower"] = float(metrics["bin_lower"][i].detach().cpu().item())
        output[f"{prefix}bin_{i:02d}_upper"] = float(metrics["bin_upper"][i].detach().cpu().item())
        output[f"{prefix}bin_{i:02d}_count"] = int(metrics["bin_count"][i].detach().cpu().item())
        output[f"{prefix}bin_{i:02d}_proportion"] = float(metrics["bin_proportion"][i].detach().cpu().item())

        acc = float(metrics["bin_accuracy"][i].detach().cpu().item())
        conf = float(metrics["bin_confidence"][i].detach().cpu().item())
        gap = float(metrics["bin_gap"][i].detach().cpu().item())

        output[f"{prefix}bin_{i:02d}_acc"] = acc
        output[f"{prefix}bin_{i:02d}_conf"] = conf
        output[f"{prefix}bin_{i:02d}_gap"] = gap

    return output


class CalibrationMetric:
    """
    Stateful calibration metric accumulator.

    This class only computes:
        - ECE
        - MCE
        - NLL
        - Brier score
        - reliability diagram statistics

    It does not compute:
        - mIoU / F1
        - segmentation metrics
        - evidence efficacy

    Usage:
        metric = CalibrationMetric(
            n_bins=15,
            input_type="logits",
            ignore_index=255,
        )

        for images, target in loader:
            logits = model(images)
            metric.update(logits, target)

        results = metric.compute()
        log_dict = metric.compute_flatten(prefix="val_")
    """

    def __init__(
        self,
        n_bins: int = 15,
        input_type: str = "auto",
        ignore_index: Optional[int] = None,
        eps: float = 1e-12,
        device: Optional[torch.device] = None,
    ):
        self.n_bins = n_bins
        self.input_type = input_type
        self.ignore_index = ignore_index
        self.eps = eps
        self.device = device

        self.reset()

    def reset(self) -> None:
        device = self.device

        self.bin_count = torch.zeros(
            self.n_bins,
            dtype=torch.long,
            device=device,
        )
        self.bin_conf_sum = torch.zeros(
            self.n_bins,
            dtype=torch.float32,
            device=device,
        )
        self.bin_correct_sum = torch.zeros(
            self.n_bins,
            dtype=torch.float32,
            device=device,
        )

        self.nll_sum = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=device,
        )
        self.brier_sum = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=device,
        )
        self.num_samples = torch.tensor(
            0,
            dtype=torch.long,
            device=device,
        )

    def update(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        probs, target_flat = flatten_probs_and_target(
            scores=scores,
            target=target,
            input_type=self.input_type,
            ignore_index=self.ignore_index,
            eps=self.eps,
        )

        if self.device is None:
            self.device = probs.device

        probs = probs.to(self.device)
        target_flat = target_flat.to(self.device)

        n = probs.size(0)

        if n == 0:
            return

        confidence, prediction = torch.max(probs, dim=1)
        correctness = prediction.eq(target_flat).float()

        confidence = confidence.clamp(min=0.0, max=1.0)
        bin_index = torch.clamp(
            (confidence * self.n_bins).long(),
            max=self.n_bins - 1,
        )

        batch_bin_count = torch.bincount(
            bin_index,
            minlength=self.n_bins,
        ).to(device=self.device)

        batch_bin_conf_sum = torch.zeros(
            self.n_bins,
            dtype=torch.float32,
            device=self.device,
        )
        batch_bin_correct_sum = torch.zeros(
            self.n_bins,
            dtype=torch.float32,
            device=self.device,
        )

        batch_bin_conf_sum.scatter_add_(0, bin_index, confidence.float())
        batch_bin_correct_sum.scatter_add_(0, bin_index, correctness.float())

        self.bin_count += batch_bin_count
        self.bin_conf_sum += batch_bin_conf_sum
        self.bin_correct_sum += batch_bin_correct_sum

        p_true = probs[
            torch.arange(n, device=self.device),
            target_flat,
        ]
        nll = -torch.log(p_true.clamp_min(self.eps))

        one_hot = F.one_hot(
            target_flat,
            num_classes=probs.size(1),
        ).float()

        brier = torch.sum((probs - one_hot) ** 2, dim=1)

        self.nll_sum += nll.sum()
        self.brier_sum += brier.sum()
        self.num_samples += n

    def compute(self) -> Dict[str, torch.Tensor]:
        device = self.bin_count.device
        eps = self.eps

        total = self.num_samples.float()

        boundaries = torch.linspace(
            0.0,
            1.0,
            self.n_bins + 1,
            device=device,
        )
        bin_lower = boundaries[:-1]
        bin_upper = boundaries[1:]

        nonempty = self.bin_count > 0

        bin_accuracy = torch.full(
            (self.n_bins,),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        bin_confidence = torch.full(
            (self.n_bins,),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

        bin_accuracy[nonempty] = (
            self.bin_correct_sum[nonempty]
            / self.bin_count[nonempty].float().clamp_min(eps)
        )
        bin_confidence[nonempty] = (
            self.bin_conf_sum[nonempty]
            / self.bin_count[nonempty].float().clamp_min(eps)
        )

        bin_gap = torch.abs(bin_accuracy - bin_confidence)
        bin_gap = torch.where(
            nonempty,
            bin_gap,
            torch.zeros_like(bin_gap),
        )

        if total <= 0:
            ece = torch.tensor(0.0, dtype=torch.float32, device=device)
            mce = torch.tensor(0.0, dtype=torch.float32, device=device)
            nll = torch.tensor(0.0, dtype=torch.float32, device=device)
            brier = torch.tensor(0.0, dtype=torch.float32, device=device)
            bin_proportion = torch.zeros(
                self.n_bins,
                dtype=torch.float32,
                device=device,
            )
        else:
            bin_proportion = self.bin_count.float() / total.clamp_min(eps)
            ece = torch.sum(bin_proportion * bin_gap)

            if nonempty.sum() == 0:
                mce = torch.tensor(0.0, dtype=torch.float32, device=device)
            else:
                mce = bin_gap[nonempty].max()

            nll = self.nll_sum / total.clamp_min(eps)
            brier = self.brier_sum / total.clamp_min(eps)

        return {
            "ECE": ece,
            "MCE": mce,
            "NLL": nll,
            "Brier": brier,
            "num_samples": self.num_samples.clone(),
            "n_bins": torch.tensor(
                self.n_bins,
                dtype=torch.long,
                device=device,
            ),
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "bin_count": self.bin_count.clone(),
            "bin_proportion": bin_proportion,
            "bin_accuracy": bin_accuracy,
            "bin_confidence": bin_confidence,
            "bin_gap": bin_gap,
        }

    def compute_python(self) -> Dict[str, Union[float, int, torch.Tensor]]:
        return metrics_to_python(self.compute())

    def compute_flatten(
        self,
        prefix: str = "",
        include_bins: bool = True,
    ) -> Dict[str, Union[float, int]]:
        return flatten_calibration_metrics(
            metrics=self.compute(),
            prefix=prefix,
            include_bins=include_bins,
        )

    def reliability_table(self) -> Dict[str, torch.Tensor]:
        return reliability_diagram_table(self.compute())

    def to(self, device: torch.device) -> "CalibrationMetric":
        self.device = device
        self.bin_count = self.bin_count.to(device)
        self.bin_conf_sum = self.bin_conf_sum.to(device)
        self.bin_correct_sum = self.bin_correct_sum.to(device)
        self.nll_sum = self.nll_sum.to(device)
        self.brier_sum = self.brier_sum.to(device)
        self.num_samples = self.num_samples.to(device)

        return self


__all__ = [
    "flatten_probs_and_target",
    "reliability_diagram_stats_from_probs",
    "calibration_error_from_bin_stats",
    "nll_from_probs",
    "brier_score_from_probs",
    "compute_calibration_metrics_from_probs",
    "compute_calibration_metrics",
    "reliability_diagram_table",
    "metrics_to_python",
    "flatten_calibration_metrics",
    "CalibrationMetric",
]