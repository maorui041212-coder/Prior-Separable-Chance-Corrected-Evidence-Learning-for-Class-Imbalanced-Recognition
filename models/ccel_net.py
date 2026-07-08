"""
CCEL-Net wrapper.

It does not define a new backbone from scratch. Instead, it wraps an existing
classification or segmentation feature extractor and replaces the normal final
prediction head with:
    evidence logits e
    optional prototype evidence scores s_hat
    prior logits b
    final logits z = b + e

Important safeguards
--------------------
1. Validation/test never uses target-derived batch prior by default.
2. Warm-up can be true ordinary CE by calling
       forward(..., use_prior=False, use_prototype=False)
   so logits = evidence_logits_linear, not b + e and not e + prototype.
3. Segmentation logits can be resized either to target spatial size or to an
   explicit output_size, which is useful for target-free inference.
4. Learnable prototype_eta is constrained to be non-negative by softplus.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F

from .evidence_branch import EvidenceHead
from .prior_branch import PriorBranch
from .prototype_memory import PrototypeMemory

Tensor = torch.Tensor


def _inverse_softplus(x: float) -> float:
    """Return y such that softplus(y) ~= x."""
    x = float(max(x, 1e-8))
    return math.log(math.expm1(x))


class CCELNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        class_prior,
        task: Literal["classification", "segmentation"] = "segmentation",
        feature_key: Optional[str] = None,
        use_prior_branch: bool = True,
        use_prototype_memory: bool = False,
        prototype_momentum: float = 0.99,
        prototype_lambda_p: float = 0.1,
        prototype_eta: float = 0.1,
        learnable_eta: bool = True,
        max_prototype_eta: Optional[float] = 5.0,
        ignore_index: Optional[int] = 255,
        prior_kwargs: Optional[Dict[str, Any]] = None,
        resize_logits_to_target: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.task = task
        self.feature_key = feature_key
        self.num_classes = int(num_classes)
        self.use_prior_branch = bool(use_prior_branch)
        self.use_prototype_memory = bool(use_prototype_memory)
        self.ignore_index = ignore_index
        self.resize_logits_to_target = bool(resize_logits_to_target)
        self.max_prototype_eta = max_prototype_eta

        self.evidence_head = EvidenceHead(feature_dim, num_classes, task=task)

        prior_kwargs = dict(prior_kwargs or {})
        self.prior_branch = PriorBranch(
            num_classes=num_classes,
            class_prior=class_prior,
            ignore_index=ignore_index,
            **prior_kwargs,
        )

        if use_prototype_memory:
            self.prototype_memory = PrototypeMemory(
                num_classes=num_classes,
                feature_dim=feature_dim,
                momentum=prototype_momentum,
                lambda_p=prototype_lambda_p,
                ignore_index=ignore_index,
            )
            if learnable_eta:
                raw = torch.tensor(_inverse_softplus(float(prototype_eta)), dtype=torch.float32)
                self.prototype_eta_raw = nn.Parameter(raw)
                self.register_buffer("prototype_eta_fixed", torch.empty(0), persistent=False)
            else:
                self.register_parameter("prototype_eta_raw", None)
                self.register_buffer(
                    "prototype_eta_fixed",
                    torch.tensor(float(prototype_eta), dtype=torch.float32),
                )
        else:
            self.prototype_memory = None
            self.register_parameter("prototype_eta_raw", None)
            self.register_buffer("prototype_eta_fixed", torch.tensor(0.0, dtype=torch.float32))

    def _extract_features(self, x: Tensor) -> Tensor:
        out = self.backbone(x)

        if isinstance(out, dict):
            if self.feature_key is not None:
                if self.feature_key not in out:
                    raise KeyError(f"feature_key={self.feature_key!r} not found in backbone output")
                return out[self.feature_key]
            for key in ("features", "feature", "feat", "last_hidden_state", "out"):
                if key in out:
                    return out[key]
            raise KeyError("backbone returned dict, but no feature_key was provided/found")

        if isinstance(out, (list, tuple)):
            return out[-1]

        return out

    def _resolve_output_size(
        self,
        target: Optional[Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Resolve segmentation output size.

        Priority:
            1. explicit output_size, useful for target-free validation/test;
            2. target spatial size, useful for training;
            3. None, keep feature/logit resolution.
        """
        if self.task != "segmentation":
            return None

        if output_size is not None:
            if len(output_size) != 2:
                raise ValueError("output_size must be a tuple/list (H, W)")
            return int(output_size[0]), int(output_size[1])

        if target is None or not self.resize_logits_to_target:
            return None
        if target.dim() != 3:
            raise ValueError(f"segmentation target must be [B,H,W], got {tuple(target.shape)}")
        return int(target.shape[-2]), int(target.shape[-1])

    @staticmethod
    def _resize_logits(logits: Tensor, output_size: Optional[Tuple[int, int]]) -> Tensor:
        if output_size is not None and logits.dim() == 4 and logits.shape[-2:] != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

    def _resize_target_to_features(self, target: Tensor, features: Tensor) -> Tensor:
        if self.task != "segmentation":
            return target
        if features.dim() != 4 or target.dim() != 3:
            return target
        if features.shape[-2:] == target.shape[-2:]:
            return target
        # Nearest-neighbor downsampling preserves integer labels.
        return F.interpolate(
            target.unsqueeze(1).float(),
            size=features.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()

    def _prototype_eta_value(self) -> Tensor:
        if self.prototype_eta_raw is not None:
            eta = F.softplus(self.prototype_eta_raw)
        else:
            eta = self.prototype_eta_fixed
        if self.max_prototype_eta is not None:
            eta = eta.clamp(max=float(self.max_prototype_eta))
        return eta

    def forward(
        self,
        x: Tensor,
        target: Optional[Tensor] = None,
        *,
        output_size: Optional[Tuple[int, int]] = None,
        update_pred_prior: bool = True,
        use_prior: Optional[bool] = None,
        use_prototype: Optional[bool] = None,
        allow_target_prior: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            target:
                Optional labels. During training it can be used for resizing and,
                only if allow_target_prior=True, for batch-prior estimation.
                During eval, target is never used by PriorBranch by default.
            output_size:
                Explicit segmentation output size (H, W). Use this for inference
                without target when the backbone outputs low-resolution features
                but final logits should match the original image size.
            update_pred_prior:
                Whether to update prediction-prior EMA in training.
            use_prior:
                Whether final logits include prior logits. For Stage-1 ordinary
                CE warm-up, set use_prior=False.
            use_prototype:
                Whether evidence logits include prototype scores. For strict
                ordinary CE warm-up, set use_prototype=False, especially when
                use_prototype_memory=True.
            allow_target_prior:
                Whether PriorBranch may use target-derived batch prior. Defaults
                to self.training. Set False for validation/test.
        """
        features = self._extract_features(x)
        resolved_output_size = self._resolve_output_size(target=target, output_size=output_size)

        # Linear/convolutional evidence head. This is the pure evidence branch
        # used for strict warm-up when use_prior=False and use_prototype=False.
        evidence_logits_linear = self.evidence_head(
            features,
            output_size=resolved_output_size,
        )

        if use_prototype is None:
            use_prototype = self.use_prototype_memory
        use_prototype = bool(use_prototype and self.use_prototype_memory and self.prototype_memory is not None)

        if use_prototype:
            proto_scores = self.prototype_memory.scores(
                features,
                pred_prior=self.prior_branch.pred_prior_ema.detach(),
            )
            proto_scores = self._resize_logits(proto_scores, resolved_output_size)
            eta = self._prototype_eta_value().to(
                device=evidence_logits_linear.device,
                dtype=evidence_logits_linear.dtype,
            )
            evidence_logits = evidence_logits_linear + eta * proto_scores
        else:
            proto_scores = torch.zeros_like(evidence_logits_linear)
            eta = self._prototype_eta_value().to(
                device=evidence_logits_linear.device,
                dtype=evidence_logits_linear.dtype,
            )
            evidence_logits = evidence_logits_linear

        if use_prior is None:
            use_prior = self.use_prior_branch
        use_prior = bool(use_prior and self.use_prior_branch)

        if allow_target_prior is None:
            allow_target_prior = bool(self.training)

        if use_prior:
            prior_logits, prior_info = self.prior_branch(
                logits_shape=tuple(evidence_logits.shape),
                target=target,
                allow_target_prior=bool(allow_target_prior),
                device=evidence_logits.device,
                dtype=evidence_logits.dtype,
            )
        else:
            prior_logits = torch.zeros_like(evidence_logits)
            prior_info = {
                "global_prior": self.prior_branch.global_prior.detach().to(evidence_logits.device),
                "pred_prior_ema": self.prior_branch.pred_prior_ema.detach().to(evidence_logits.device),
                "chance_prior_for_loss": self.prior_branch.global_prior.detach().to(evidence_logits.device),
                "batch_prior_available": torch.tensor(False, device=evidence_logits.device),
            }

        logits = evidence_logits + prior_logits
        prob = F.softmax(logits, dim=1)
        evidence_prob = F.softmax(evidence_logits, dim=1)
        evidence_prob_linear = F.softmax(evidence_logits_linear, dim=1)

        if self.training and update_pred_prior and self.use_prior_branch:
            # Warm-up with use_prior=False/use_prototype=False updates EMA from
            # pure evidence predictions. Decoupled stages update from final logits.
            self.prior_branch.update_pred_prior(prob.detach())

        return {
            "logits": logits,
            "prob": prob,
            "evidence_logits": evidence_logits,
            "evidence_logits_linear": evidence_logits_linear,
            "evidence_prob": evidence_prob,
            "evidence_prob_linear": evidence_prob_linear,
            "prior_logits": prior_logits,
            "proto_scores": proto_scores,
            "prototype_eta": eta.detach(),
            "used_prior": torch.tensor(use_prior, device=logits.device),
            "used_prototype": torch.tensor(use_prototype, device=logits.device),
            "features": features,
            "prior_info": prior_info,
        }

    @torch.no_grad()
    def update_prototypes(
        self,
        features: Tensor,
        target: Tensor,
        psi: Optional[Tensor] = None,
    ) -> None:
        """
        Update prototype memory. This is intentionally not called inside forward,
        because prototype updates are non-gradient state updates and should be
        controlled by the trainer after optimizer.step().
        """
        if self.use_prototype_memory and self.prototype_memory is not None:
            proto_target = self._resize_target_to_features(target, features)
            self.prototype_memory.update(features=features, target=proto_target, psi=psi)
