from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def normalize_prior(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float().clamp_min(0)
    return x / x.sum().clamp_min(eps)


@torch.no_grad()
def prior_l1(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = normalize_prior(p)
    q = normalize_prior(q)
    return torch.abs(p - q).sum()


@torch.no_grad()
def prior_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = normalize_prior(p, eps)
    q = normalize_prior(q, eps)
    return torch.sum(p * torch.log(p.clamp_min(eps) / q.clamp_min(eps)))


@torch.no_grad()
def target_prior_from_labels(
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    target = target.reshape(-1).long()

    if ignore_index is not None:
        target = target[target != int(ignore_index)]

    valid = (target >= 0) & (target < num_classes)
    target = target[valid]

    if target.numel() == 0:
        return torch.zeros(num_classes, device=target.device)

    counts = torch.bincount(target, minlength=num_classes).float()
    return normalize_prior(counts)


@torch.no_grad()
def pred_prior_from_probs(
    probs: torch.Tensor,
    ignore_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Classification:
        probs: [B, C]

    Segmentation:
        probs: [B, C, H, W]
    """
    if probs.dim() == 2:
        return normalize_prior(probs.mean(dim=0))

    if probs.dim() == 4:
        if ignore_mask is not None:
            # ignore_mask: [B, H, W], True means valid
            probs = probs.permute(0, 2, 3, 1)
            probs = probs[ignore_mask]
            if probs.numel() == 0:
                return torch.zeros(probs.shape[-1], device=probs.device)
            return normalize_prior(probs.mean(dim=0))

        return normalize_prior(probs.mean(dim=(0, 2, 3)))

    raise ValueError(f"Unsupported probs shape: {tuple(probs.shape)}")


@torch.no_grad()
def prior_shift_diagnostics(
    probs: torch.Tensor,
    target: torch.Tensor,
    train_prior: torch.Tensor,
    num_classes: int,
    evidence_probs: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    if probs.dim() == 4 and ignore_index is not None:
        ignore_mask = target != int(ignore_index)
    else:
        ignore_mask = None

    test_prior = target_prior_from_labels(
        target=target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    pred_prior = pred_prior_from_probs(
        probs=probs,
        ignore_mask=ignore_mask,
    )

    out = {
        "pred_to_train_l1": float(prior_l1(pred_prior, train_prior).cpu()),
        "pred_to_test_l1": float(prior_l1(pred_prior, test_prior).cpu()),
        "pred_to_train_kl": float(prior_kl(pred_prior, train_prior).cpu()),
        "pred_to_test_kl": float(prior_kl(pred_prior, test_prior).cpu()),
    }

    for c in range(num_classes):
        out[f"test_prior_class{c}"] = float(test_prior[c].cpu())
        out[f"pred_prior_class{c}"] = float(pred_prior[c].cpu())
        out[f"train_prior_class{c}"] = float(normalize_prior(train_prior)[c].cpu())

    if evidence_probs is not None:
        evi_prior = pred_prior_from_probs(
            probs=evidence_probs,
            ignore_mask=ignore_mask,
        )
        out["evidence_to_train_l1"] = float(prior_l1(evi_prior, train_prior).cpu())
        out["evidence_to_test_l1"] = float(prior_l1(evi_prior, test_prior).cpu())

        for c in range(num_classes):
            out[f"evidence_prior_class{c}"] = float(evi_prior[c].cpu())

    return out


def compute_shift_drop(
    id_metrics: Dict[str, float],
    shift_metrics: Dict[str, float],
    metric_names,
) -> Dict[str, float]:
    out = {}
    for name in metric_names:
        if name not in id_metrics or name not in shift_metrics:
            continue
        out[f"drop_{name}"] = float(id_metrics[name] - shift_metrics[name])
    return out


def compute_shift_increase(
    id_metrics: Dict[str, float],
    shift_metrics: Dict[str, float],
    metric_names,
) -> Dict[str, float]:
    out = {}
    for name in metric_names:
        if name not in id_metrics or name not in shift_metrics:
            continue
        out[f"increase_{name}"] = float(shift_metrics[name] - id_metrics[name])
    return out