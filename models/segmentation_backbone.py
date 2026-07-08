"""
Prior branch for Prior-Separable Chance-Corrected Evidence Learning (CCEL).

This module corresponds to the prior-induced bias term in the paper:

    z_i = b_i + e_i,
    b_i = g(log pi_global, log pi_batch, log r_bar).

It estimates only distribution-level bias logits, not input-dependent visual
or semantic evidence. The evidence branch is therefore evaluated separately by
the chance-corrected evidence efficacy constraint.

Important implementation details
--------------------------------
1. Validation/test must not use target-derived batch prior. This branch only uses
   batch prior when ``allow_target_prior=True`` and target is provided. In CCEL,
   the default is allow_target_prior = model.training.
2. When target is unavailable, batch prior is skipped instead of being replaced by
   global prior. This avoids counting the same global prior twice.
3. Multiple prior terms can become too strong under extreme imbalance. Therefore,
   enabled terms can be weight-normalized and the final prior logits can be scaled
   and clipped. The returned ``info`` dictionary exposes each prior component for
   paper-consistent logging and ablation.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

Tensor = torch.Tensor


class PriorBranch(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_prior,
        eps: float = 1e-6,
        kappa: float = 32.0,
        beta: float = 0.99,
        use_global_prior: bool = True,
        use_batch_prior: bool = True,
        use_pred_prior_ema: bool = True,
        global_prior_weight: float = 1.0,
        batch_prior_weight: float = 1.0,
        pred_prior_weight: float = 1.0,
        normalize_prior_weights: bool = True,
        prior_logit_scale: float = 1.0,
        max_abs_prior_logit: Optional[float] = 6.0,
        center_logits: bool = True,
        ignore_index: Optional[int] = 255,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        self.kappa = float(kappa)
        self.beta = float(beta)
        self.use_global_prior = bool(use_global_prior)
        self.use_batch_prior = bool(use_batch_prior)
        self.use_pred_prior_ema = bool(use_pred_prior_ema)
        self.global_prior_weight = float(global_prior_weight)
        self.batch_prior_weight = float(batch_prior_weight)
        self.pred_prior_weight = float(pred_prior_weight)
        self.normalize_prior_weights = bool(normalize_prior_weights)
        self.prior_logit_scale = float(prior_logit_scale)
        self.max_abs_prior_logit = max_abs_prior_logit
        self.center_logits = bool(center_logits)
        self.ignore_index = ignore_index

        prior = torch.as_tensor(class_prior, dtype=torch.float32).reshape(-1)
        if prior.numel() != self.num_classes:
            raise ValueError(f"class_prior must have {self.num_classes} elements")
        prior = prior.clamp_min(self.eps)
        prior = prior / prior.sum().clamp_min(self.eps)

        self.register_buffer("global_prior", prior.clone())
        self.register_buffer("pred_prior_ema", prior.clone())

    @torch.no_grad()
    def _target_counts(self, target: Tensor) -> Tensor:
        """Count valid labels in target, ignoring ignore_index and invalid labels."""
        flat = target.reshape(-1).long()
        if self.ignore_index is not None:
            flat = flat[flat != int(self.ignore_index)]
        flat = flat[(flat >= 0) & (flat < self.num_classes)]
        if flat.numel() == 0:
            return torch.zeros(self.num_classes, device=target.device, dtype=torch.float32)
        return torch.bincount(flat, minlength=self.num_classes).to(device=target.device, dtype=torch.float32)

    @torch.no_grad()
    def batch_prior(self, target: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute shrunk batch prior.

        pi_tilde_c = (1 - alpha_c) * pi_global_c + alpha_c * pi_batch_c
        alpha_c    = n_c / (n_c + kappa)

        Returns:
            pi_tilde: [C]
            pi_batch: [C]
            counts:   [C]
        """
        counts = self._target_counts(target)
        total = counts.sum().clamp_min(1.0)
        pi_batch = counts / total
        alpha = counts / (counts + self.kappa)

        global_prior = self.global_prior.to(device=target.device, dtype=torch.float32)
        pi_tilde = (1.0 - alpha) * global_prior + alpha * pi_batch
        pi_tilde = pi_tilde.clamp_min(self.eps)
        pi_tilde = pi_tilde / pi_tilde.sum().clamp_min(self.eps)
        return pi_tilde, pi_batch, counts

    @torch.no_grad()
    def update_pred_prior(self, probs: Tensor) -> Tensor:
        """Update EMA of model prediction prior using final probabilities."""
        if probs.dim() == 2:
            r = probs.detach().mean(dim=0)
        elif probs.dim() == 4:
            r = probs.detach().mean(dim=(0, 2, 3))
        else:
            raise ValueError(f"probs must be [B,C] or [B,C,H,W], got {tuple(probs.shape)}")

        r = r.to(device=self.pred_prior_ema.device, dtype=self.pred_prior_ema.dtype)
        r = r.clamp_min(self.eps)
        r = r / r.sum().clamp_min(self.eps)
        self.pred_prior_ema.mul_(self.beta).add_(r, alpha=1.0 - self.beta)
        self.pred_prior_ema.div_(self.pred_prior_ema.sum().clamp_min(self.eps))
        return self.pred_prior_ema

    def _broadcast(self, prior_logits: Tensor, logits_shape: Tuple[int, ...]) -> Tensor:
        if len(logits_shape) == 2:
            bsz, c = logits_shape
            return prior_logits.view(1, c).expand(bsz, c)
        if len(logits_shape) == 4:
            bsz, c, h, w = logits_shape
            return prior_logits.view(1, c, 1, 1).expand(bsz, c, h, w)
        raise ValueError(f"logits_shape must be [B,C] or [B,C,H,W], got {logits_shape}")

    def _add_prior_term(self, terms, weights, names, prior: Tensor, weight: float, name: str) -> None:
        if weight == 0.0:
            return
        terms.append(torch.log(prior.clamp_min(self.eps)))
        weights.append(float(weight))
        names.append(str(name))

    def forward(
        self,
        logits_shape: Tuple[int, ...],
        target: Optional[Tensor] = None,
        *,
        allow_target_prior: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Args:
            logits_shape:
                Shape of evidence logits, [B,C] or [B,C,H,W].
            target:
                Ground truth labels. Only used when allow_target_prior=True.
            allow_target_prior:
                Whether target-derived batch prior is allowed. Set False during
                validation/test to avoid label leakage.
            device, dtype:
                Output device/dtype. Usually evidence_logits.device/dtype.
        """
        device = device or self.global_prior.device
        dtype = dtype or self.global_prior.dtype

        global_prior = self.global_prior.to(device=device, dtype=dtype).clamp_min(self.eps)
        global_prior = global_prior / global_prior.sum().clamp_min(self.eps)
        pred_prior = self.pred_prior_ema.to(device=device, dtype=dtype).clamp_min(self.eps)
        pred_prior = pred_prior / pred_prior.sum().clamp_min(self.eps)

        batch_prior_available = bool(target is not None and allow_target_prior and self.use_batch_prior)
        pi_batch_raw = torch.empty(0, device=device, dtype=dtype)
        batch_counts = torch.empty(0, device=device, dtype=dtype)

        terms = []
        weights = []
        names = []

        log_global_prior = torch.log(global_prior.clamp_min(self.eps))
        log_pred_prior = torch.log(pred_prior.clamp_min(self.eps))
        log_batch_prior = torch.empty(0, device=device, dtype=dtype)

        if self.use_global_prior:
            self._add_prior_term(terms, weights, names, global_prior, self.global_prior_weight, "global")

        if batch_prior_available:
            batch_prior, pi_batch_raw, batch_counts = self.batch_prior(target)
            batch_prior = batch_prior.to(device=device, dtype=dtype)
            pi_batch_raw = pi_batch_raw.to(device=device, dtype=dtype)
            batch_counts = batch_counts.to(device=device, dtype=dtype)
            log_batch_prior = torch.log(batch_prior.clamp_min(self.eps))
            self._add_prior_term(terms, weights, names, batch_prior, self.batch_prior_weight, "batch")
        else:
            batch_prior = torch.empty(0, device=device, dtype=dtype)

        if self.use_pred_prior_ema:
            self._add_prior_term(terms, weights, names, pred_prior, self.pred_prior_weight, "pred")

        if not terms:
            prior_logits = torch.zeros(self.num_classes, device=device, dtype=dtype)
            effective_weight_sum = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            weight_tensor = torch.tensor(weights, device=device, dtype=dtype)
            stacked = torch.stack(terms, dim=0)
            if self.normalize_prior_weights:
                denom = weight_tensor.abs().sum().clamp_min(self.eps)
                prior_logits = (stacked * weight_tensor.view(-1, 1)).sum(dim=0) / denom
                effective_weight_sum = denom.detach()
            else:
                prior_logits = (stacked * weight_tensor.view(-1, 1)).sum(dim=0)
                effective_weight_sum = weight_tensor.abs().sum().detach()

        raw_prior_logits = prior_logits

        if self.center_logits:
            prior_logits = prior_logits - prior_logits.mean()

        scaled_prior_logits = prior_logits * self.prior_logit_scale
        prior_logits = scaled_prior_logits

        if self.max_abs_prior_logit is not None:
            prior_logits = prior_logits.clamp(
                min=-float(self.max_abs_prior_logit),
                max=float(self.max_abs_prior_logit),
            )

        broadcast_prior_logits = self._broadcast(prior_logits, logits_shape)

        # chance_prior_for_loss is the best available training baseline. During
        # eval/test with no target prior, it falls back to global prior and does
        # not leak labels.
        if batch_prior_available:
            chance_prior_for_loss = batch_prior.detach()
        else:
            chance_prior_for_loss = global_prior.detach()

        info: Dict[str, Tensor] = {
            # Priors in paper notation.
            "global_prior": global_prior.detach(),
            "pi_global": global_prior.detach(),
            "pred_prior_ema": pred_prior.detach(),
            "r_bar": pred_prior.detach(),
            "chance_prior_for_loss": chance_prior_for_loss,
            "pi_tilde": chance_prior_for_loss,

            # Bias logits b_i and component logs before broadcasting.
            "prior_logits_vector": prior_logits.detach(),
            "b_vector": prior_logits.detach(),
            "raw_prior_logits_vector": raw_prior_logits.detach(),
            "scaled_prior_logits_vector": scaled_prior_logits.detach(),
            "log_global_prior": log_global_prior.detach(),
            "log_pred_prior": log_pred_prior.detach(),
            "log_batch_prior": log_batch_prior.detach(),

            # Diagnostics for ablation and reproducibility.
            "batch_prior_available": torch.tensor(batch_prior_available, device=device),
            "effective_prior_weight_sum": effective_weight_sum,
            "enabled_prior_terms": torch.tensor(len(terms), device=device, dtype=dtype),
        }
        if batch_prior_available:
            info.update(
                {
                    "batch_prior": batch_prior.detach(),
                    "pi_batch_tilde": batch_prior.detach(),
                    "batch_prior_raw": pi_batch_raw.detach(),
                    "pi_batch_raw": pi_batch_raw.detach(),
                    "batch_counts": batch_counts.detach(),
                }
            )
        return broadcast_prior_logits, info
