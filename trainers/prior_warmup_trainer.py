"""Stage-1 Prior Warm-up trainer for CCEL-Net.

The intended warm-up in CCEL-Net is strict ordinary CE:

    L = CE(e_linear, y)

Therefore this trainer always forwards CCELNet with:
    use_prior=False
    use_prototype=False
    allow_target_prior=False

Prediction-prior EMA can still be updated from pure evidence predictions, and
prototype memory can optionally be initialized after optimizer.step(), but neither
prior logits nor prototype scores participate in the warm-up logits.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
from torch import nn

from .base_trainer import BaseTrainer, Tensor


class PriorWarmupTrainer(BaseTrainer):
    """Strict CE warm-up trainer.

    Args:
        task:
            "segmentation" or "classification". Segmentation logits are resized
            to target size before CE.
        update_pred_prior:
            If True, CCELNet updates r_bar from pure evidence predictions during
            warm-up. This matches the idea of collecting model prediction prior.
        initialize_prototypes:
            If True, update prototype memory after optimizer.step(). Prototypes do
            not affect warm-up logits because use_prototype=False.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        task: Literal["segmentation", "classification"] = "segmentation",
        ignore_index: Optional[int] = 255,
        ce_weight: Optional[Tensor] = None,
        update_pred_prior: bool = True,
        initialize_prototypes: bool = True,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(model, optimizer, **base_kwargs)
        if task not in {"segmentation", "classification"}:
            raise ValueError("task must be 'segmentation' or 'classification'")
        self.task = task
        self.ignore_index = ignore_index if task == "segmentation" else None
        self.ce_weight = ce_weight.to(self.device) if torch.is_tensor(ce_weight) else ce_weight
        self.update_pred_prior = bool(update_pred_prior)
        self.initialize_prototypes = bool(initialize_prototypes)

    def forward_model(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        training: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        output_size = tuple(targets.shape[-2:]) if self.task == "segmentation" else None
        try:
            return self.model(
                images,
                target=targets,
                output_size=output_size,
                update_pred_prior=self.update_pred_prior and training,
                use_prior=False,
                use_prototype=False,
                allow_target_prior=False,
            )
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword" not in msg and "got an unexpected" not in msg:
                raise
            logits = self.model(images)
            if torch.is_tensor(logits):
                return {"logits": logits, "evidence_logits_linear": logits}
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
        # Prefer the pure evidence head for clarity. In strict warm-up, logits and
        # evidence_logits_linear should be identical for CCELNet.
        logits = outputs.get("evidence_logits_linear", outputs["logits"])
        loss = self._call_ce(
            logits,
            targets,
            ignore_index=self.ignore_index,
            weight=self.ce_weight,
        )
        return {"loss": loss, "ce_loss": loss.detach(), "warmup_ce_loss": loss.detach()}

    def after_optimizer_step(
        self,
        outputs: Dict[str, Tensor],
        loss_dict: Dict[str, Tensor],
        targets: Tensor,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.initialize_prototypes:
            return
        if not hasattr(self.model, "update_prototypes"):
            return
        if "features" not in outputs:
            return
        self.model.update_prototypes(
            features=outputs["features"].detach(),
            target=targets,
            psi=None,
        )
