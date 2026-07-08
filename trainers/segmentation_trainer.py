"""Segmentation trainer for CCEL-Net.

This trainer is intended for the Evidence Decoupling stage or CE-only
segmentation baselines. It can train:
    - ordinary segmentation networks returning logits;
    - CCELNet with z = b + e, by setting use_prior=True;
    - CCELNet without primal-dual efficacy constraints.

For primal-dual training, use primal_dual_trainer.py instead.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .base_trainer import BaseTrainer, Tensor


class SegmentationTrainer(BaseTrainer):
    """Generic segmentation trainer.

    Args:
        criterion:
            Optional loss. If None, use standard CE on outputs["logits"].
            If provided, it may accept either (outputs, target) or (logits, target).
        use_prior:
            Whether CCELNet forward should include prior logits. Stage-2 evidence
            decoupling usually sets True. CE baseline/warm-up sets False.
        use_prototype:
            Whether CCELNet forward should include prototype scores.
        allow_target_prior_train:
            True only for training. Validation/test always use False to avoid
            target-prior leakage.
        update_prototypes:
            If True, call model.update_prototypes after optimizer.step(). This is
            off by default here; stage-3 usually uses PrimalDualTrainer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        criterion: Optional[nn.Module] = None,
        ignore_index: Optional[int] = 255,
        ce_weight: Optional[Tensor] = None,
        use_prior: bool = True,
        use_prototype: bool = False,
        allow_target_prior_train: bool = True,
        update_pred_prior: bool = True,
        update_prototypes: bool = False,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(model, optimizer, **base_kwargs)
        self.criterion = criterion
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight.to(self.device) if torch.is_tensor(ce_weight) else ce_weight
        self.use_prior = bool(use_prior)
        self.use_prototype = bool(use_prototype)
        self.allow_target_prior_train = bool(allow_target_prior_train)
        self.update_pred_prior = bool(update_pred_prior)
        self.update_prototypes = bool(update_prototypes)

    def forward_model(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        training: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        output_size = tuple(targets.shape[-2:])
        try:
            return self.model(
                images,
                target=targets,
                output_size=output_size,
                update_pred_prior=self.update_pred_prior and training,
                use_prior=self.use_prior,
                use_prototype=self.use_prototype,
                allow_target_prior=(training and self.allow_target_prior_train),
            )
        except TypeError as exc:
            # Fallback for plain segmentation networks.
            msg = str(exc)
            if "unexpected keyword" not in msg and "got an unexpected" not in msg:
                raise
            logits = self.model(images)
            if torch.is_tensor(logits):
                return {"logits": logits}
            if isinstance(logits, dict):
                return logits
            raise TypeError("Plain model must return logits tensor or dict with 'logits'")

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Tensor,
        *,
        training: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        if self.criterion is None:
            loss = self._call_ce(
                outputs["logits"],
                targets,
                ignore_index=self.ignore_index,
                weight=self.ce_weight,
            )
            return {"loss": loss, "ce_loss": loss.detach()}

        try:
            out = self.criterion(outputs, targets)
        except TypeError:
            out = self.criterion(outputs["logits"], targets)

        if torch.is_tensor(out):
            return {"loss": out}
        if isinstance(out, dict):
            return out
        raise TypeError("criterion must return a tensor or dict containing 'loss'")

    def after_optimizer_step(
        self,
        outputs: Dict[str, Tensor],
        loss_dict: Dict[str, Tensor],
        targets: Tensor,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.update_prototypes:
            return
        if not hasattr(self.model, "update_prototypes"):
            return
        if "features" not in outputs:
            return
        psi = loss_dict.get("psi_for_dual", loss_dict.get("psi", None))
        if torch.is_tensor(psi):
            psi = psi.detach()
        self.model.update_prototypes(
            features=outputs["features"].detach(),
            target=targets,
            psi=psi,
        )
