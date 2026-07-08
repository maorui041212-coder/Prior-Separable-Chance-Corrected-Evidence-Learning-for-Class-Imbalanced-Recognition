"""
Dual-guided efficacy prototype memory for CCEL.

For each class c, maintain an EMA prototype m_c in feature space.
Prototype scores are cosine similarities corrected by prediction prior:
    s_hat_ic = cosine(h_i, m_c) - lambda_p * log(r_bar_c)

Key implementation details:
1) Uninitialized classes must not receive non-zero prototype scores.
   Prior correction is applied only to initialized classes and the final scores
   are masked again after correction.
2) Prior correction can be clipped by max_abs_correction because log(r_bar_c)
   may be very large when a class has tiny predicted prior.
3) The module provides an optional forward(..., update=True) interface so a
   trainer/model can update prototypes and get scores through a single call.
   For strict warm-up CE, call CCELNet.forward(..., use_prototype=False), or call
   this module with update=False and do not add returned scores to logits.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


class PrototypeMemory(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.99,
        lambda_p: float = 0.1,
        max_abs_correction: Optional[float] = 1.0,
        eps: float = 1e-6,
        ignore_index: Optional[int] = 255,
    ) -> None:
        """
        Args:
            num_classes:
                Number of classes C.
            feature_dim:
                Feature dimension D.
            momentum:
                EMA momentum for prototypes. Larger means slower update.
            lambda_p:
                Strength of chance correction in prototype scores.
            max_abs_correction:
                Optional clamp for lambda_p * log(pred_prior). This prevents
                tiny predicted priors from dominating cosine similarity.
                Use None to disable clipping.
            eps:
                Numerical stability.
            ignore_index:
                Ignored label value for segmentation masks.
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.lambda_p = float(lambda_p)
        self.max_abs_correction = None if max_abs_correction is None else float(max_abs_correction)
        self.eps = float(eps)
        self.ignore_index = ignore_index

        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer("initialized", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("class_update_count", torch.zeros(self.num_classes, dtype=torch.long))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _resize_target_if_needed(self, target: Tensor, features: Tensor) -> Tensor:
        """Resize segmentation target to feature resolution using nearest neighbor."""
        if features.dim() == 4:
            _, _, h, w = features.shape
            if target.dim() != 3:
                raise ValueError(f"segmentation target must be [B,H,W], got {tuple(target.shape)}")
            if target.shape[-2:] != (h, w):
                target = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=(h, w),
                    mode="nearest",
                ).squeeze(1).long()
        return target

    def _flatten_features_and_target(self, features: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """Return normalized flat features [N,D] and labels [N]."""
        target = self._resize_target_if_needed(target, features)

        if features.dim() == 2:
            flat_features = features
            flat_target = target.reshape(-1).long()
        elif features.dim() == 4:
            flat_features = features.permute(0, 2, 3, 1).reshape(-1, features.size(1))
            flat_target = target.reshape(-1).long()
        else:
            raise ValueError(f"features must be [B,D] or [B,D,H,W], got {tuple(features.shape)}")

        if flat_features.size(1) != self.feature_dim:
            raise ValueError(
                f"feature dimension mismatch: expected {self.feature_dim}, got {flat_features.size(1)}"
            )

        if self.ignore_index is not None:
            valid = flat_target != int(self.ignore_index)
            flat_features = flat_features[valid]
            flat_target = flat_target[valid]

        valid_label = (flat_target >= 0) & (flat_target < self.num_classes)
        flat_features = flat_features[valid_label]
        flat_target = flat_target[valid_label]

        if flat_features.numel() == 0:
            return flat_features, flat_target

        flat_features = F.normalize(flat_features, dim=1)
        return flat_features, flat_target

    def _prototype_scores_without_correction(self, features: Tensor) -> Tensor:
        """Cosine similarity to prototypes, with uninitialized classes masked to zero."""
        if features.size(1) != self.feature_dim:
            raise ValueError(
                f"feature dimension mismatch: expected {self.feature_dim}, got {features.size(1)}"
            )

        proto = F.normalize(self.prototypes.to(device=features.device, dtype=features.dtype), dim=1)
        valid = self.initialized.to(device=features.device)

        if features.dim() == 2:
            x = F.normalize(features, dim=1)
            sim = x @ proto.t()  # [B,C]
            sim = sim.masked_fill(~valid.view(1, -1), 0.0)
            return sim

        if features.dim() == 4:
            x = F.normalize(features, dim=1)
            sim = torch.einsum("bdhw,cd->bchw", x, proto)  # [B,C,H,W]
            sim = sim.masked_fill(~valid.view(1, -1, 1, 1), 0.0)
            return sim

        raise ValueError(f"features must be [B,D] or [B,D,H,W], got {tuple(features.shape)}")

    def _prior_correction(self, pred_prior: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Return lambda_p * log(pred_prior), optionally clipped.

        This value is subtracted from cosine similarity. Since log(pred_prior)
        is negative for small priors, subtracting it boosts tail classes. Clipping
        prevents this boost from exceeding the cosine-similarity scale.
        """
        pred_prior = pred_prior.to(device=device, dtype=dtype).reshape(-1)
        if pred_prior.numel() != self.num_classes:
            raise ValueError(f"pred_prior must have shape [{self.num_classes}]")
        pred_prior = pred_prior.clamp_min(self.eps)
        pred_prior = pred_prior / pred_prior.sum().clamp_min(self.eps)

        correction = self.lambda_p * torch.log(pred_prior)
        if self.max_abs_correction is not None:
            correction = correction.clamp(
                min=-self.max_abs_correction,
                max=self.max_abs_correction,
            )
        return correction

    # ------------------------------------------------------------------
    # Prototype update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        features: Tensor,
        target: Tensor,
        psi: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Update class prototypes with efficacy-guided EMA.

        Intended rule:
            omega_c = 1 - psi_c
            lr_c = (1 - momentum) * omega_c
            m_c <- normalize((1 - lr_c) m_c + lr_c hbar_c)

        For an uninitialized class, the first observed class mean directly
        initializes the prototype, independent of omega_c.

        Returns a small update log dictionary. ``prototype_update_strength`` is
        omega_c in the paper.
        """
        flat_features, flat_target = self._flatten_features_and_target(features, target)
        device = features.device
        dtype = features.dtype

        updated_mask = torch.zeros(self.num_classes, device=device, dtype=torch.bool)
        lr_used = torch.zeros(self.num_classes, device=device, dtype=dtype)

        if flat_features.numel() == 0:
            return {
                "updated_mask": updated_mask,
                "prototype_lr": lr_used,
                "prototype_update_strength": torch.zeros(self.num_classes, device=device, dtype=dtype),
            }

        if psi is None:
            update_strength = torch.ones(self.num_classes, device=device, dtype=dtype)
        else:
            psi = psi.detach().to(device=device, dtype=dtype).reshape(-1)
            if psi.numel() != self.num_classes:
                raise ValueError(f"psi must have shape [{self.num_classes}]")
            # omega_c = 1 - psi_c. Low-efficacy classes receive stronger EMA updates.
            update_strength = (1.0 - psi).clamp(0.0, 1.0)

        prototypes = self.prototypes.to(device=device, dtype=dtype)

        for c in range(self.num_classes):
            mask = flat_target == c
            if not bool(mask.any().item()):
                continue

            hbar = flat_features[mask].mean(dim=0)
            hbar = F.normalize(hbar, dim=0)

            if not bool(self.initialized[c].item()):
                prototypes[c].copy_(hbar)
                self.initialized[c] = True
                self.class_update_count[c] += 1
                updated_mask[c] = True
                lr_used[c] = 1.0
                continue

            lr_c = (1.0 - self.momentum) * float(update_strength[c].item())
            if lr_c <= 0.0:
                continue

            new_proto = (1.0 - lr_c) * prototypes[c] + lr_c * hbar
            prototypes[c].copy_(F.normalize(new_proto, dim=0))
            self.class_update_count[c] += 1
            updated_mask[c] = True
            lr_used[c] = float(lr_c)

        # Ensure the stored buffer is updated if it had to be moved to another device/dtype.
        if self.prototypes.device != device or self.prototypes.dtype != dtype:
            self.prototypes.data = prototypes.to(device=self.prototypes.device, dtype=self.prototypes.dtype)
        else:
            self.prototypes.copy_(prototypes)

        return {
            "updated_mask": updated_mask,
            "prototype_lr": lr_used,
            "prototype_update_strength": update_strength.detach().clone(),
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def scores(
        self,
        features: Tensor,
        pred_prior: Optional[Tensor] = None,
        *,
        return_info: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Return chance-corrected prototype scores with shape [B,C] or [B,C,H,W].

        Important:
            Uninitialized classes are always forced to score 0.0, even after
            prior correction. This avoids fake prototype evidence for classes
            that have no prototype yet.
        """
        sim = self._prototype_scores_without_correction(features)
        valid = self.initialized.to(device=sim.device)

        correction = None
        if pred_prior is not None:
            correction = self._prior_correction(pred_prior, device=sim.device, dtype=sim.dtype)

            # Apply correction only to initialized classes.
            correction = correction * valid.to(dtype=sim.dtype)
            if sim.dim() == 2:
                sim = sim - correction.view(1, -1)
                sim = sim.masked_fill(~valid.view(1, -1), 0.0)
            elif sim.dim() == 4:
                sim = sim - correction.view(1, -1, 1, 1)
                sim = sim.masked_fill(~valid.view(1, -1, 1, 1), 0.0)
            else:
                raise RuntimeError(f"unexpected score shape {tuple(sim.shape)}")

        if not return_info:
            return sim

        info: Dict[str, Tensor] = {
            "initialized": valid.detach().clone(),
            "class_update_count": self.class_update_count.detach().to(device=sim.device).clone(),
            "s_hat": sim.detach().clone(),
        }
        if correction is not None:
            info["prior_correction"] = correction.detach().clone()
        return sim, info

    def forward(
        self,
        features: Tensor,
        target: Optional[Tensor] = None,
        *,
        pred_prior: Optional[Tensor] = None,
        psi: Optional[Tensor] = None,
        update: bool = False,
        update_before_score: bool = True,
        return_info: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Optional combined update-and-score interface.

        Args:
            features:
                [B,D] for classification or [B,D,H,W] for segmentation.
            target:
                Required if update=True.
            pred_prior:
                Prediction-prior EMA used for chance correction.
            psi:
                Class-wise efficacy used to control update strength.
            update:
                If True, update prototypes in this call. Use this only in
                training stages where prototype memory is supposed to be active.
            update_before_score:
                If True, update prototypes first and then compute scores.
                If False, compute scores using old prototypes and update after.
            return_info:
                If True, return (scores, info).

        Note:
            This does not make warm-up automatically use prototypes. Strict CE
            warm-up should still call CCELNet.forward(..., use_prototype=False).
        """
        update_info: Optional[Dict[str, Tensor]] = None
        if update and target is None:
            raise ValueError("target must be provided when update=True")

        if update and update_before_score:
            update_info = self.update(features=features, target=target, psi=psi)

        score_out = self.scores(features=features, pred_prior=pred_prior, return_info=return_info)

        if update and not update_before_score:
            update_info = self.update(features=features, target=target, psi=psi)

        if return_info:
            scores, info = score_out  # type: ignore[misc]
            if update_info is not None:
                info.update({f"update_{k}": v for k, v in update_info.items()})
            return scores, info

        return score_out
