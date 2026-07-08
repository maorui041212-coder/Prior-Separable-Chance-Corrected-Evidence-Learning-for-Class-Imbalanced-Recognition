"""Stage-3 primal-dual trainer for CCEL-Net.

This trainer owns the Efficacy-Guided Rebalancing stage at training-loop level:
    - forward CCELNet with prior/evidence decomposition;
    - call PrimalDualEfficacyLoss for CE + mu * violation;
    - optionally use an EMA efficacy meter for stable dual-variable updates;
    - update prototype memory explicitly after optimizer.step().

The loss object owns the mathematical objective and dual buffers. This trainer
only wires together model outputs, stable efficacy estimates, and prototype EMA.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
from torch import nn

from .base_trainer import BaseTrainer, Tensor

try:
    from ccel.metrics.efficacy_metrics import ChanceCorrectedEfficacyMeter
except Exception:  # pragma: no cover - keeps import robust during partial builds
    ChanceCorrectedEfficacyMeter = None  # type: ignore


class PrimalDualTrainer(BaseTrainer):
    """Trainer for CE(b+e) + primal-dual efficacy constraints.

    Args:
        criterion:
            A PrimalDualEfficacyLoss instance. It should accept outputs, target,
            chance_prior, use_constraint, auto_update_mu, and optional dual psi.
        task:
            "segmentation" or "classification".
        use_prior:
            Usually True for this stage.
        use_prototype:
            Set True in the full CCEL-Net stage; can be False for evidence
            decoupling with primal-dual constraint but without prototype memory.
        use_ema_for_dual:
            If True, use ChanceCorrectedEfficacyMeter to update mu from smoothed
            psi rather than noisy mini-batch psi. Recommended for imbalanced
            segmentation.
        efficacy_meter:
            Optional externally constructed meter. If None and use_ema_for_dual
            is True, the trainer tries to create one from criterion.num_classes.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        *,
        criterion: nn.Module,
        task: Literal["segmentation", "classification"] = "segmentation",
        use_prior: bool = True,
        use_prototype: bool = True,
        allow_target_prior_train: bool = True,
        update_pred_prior: bool = True,
        update_prototypes: bool = True,
        use_ema_for_dual: bool = True,
        efficacy_meter: Optional[nn.Module] = None,
        meter_momentum: float = 0.95,
        min_class_count: int = 32,
        min_valid_count: int = 256,
        warmup_updates: int = 10,
        chance_prior_for_dual: str = "ema",
        **base_kwargs: Any,
    ) -> None:
        super().__init__(model, optimizer, **base_kwargs)
        if task not in {"segmentation", "classification"}:
            raise ValueError("task must be 'segmentation' or 'classification'")
        if chance_prior_for_dual not in {"ema", "loss"}:
            raise ValueError("chance_prior_for_dual must be 'ema' or 'loss'")

        self.criterion = criterion.to(self.device)
        self.task = task
        self.use_prior = bool(use_prior)
        self.use_prototype = bool(use_prototype)
        self.allow_target_prior_train = bool(allow_target_prior_train)
        self.update_pred_prior = bool(update_pred_prior)
        self.update_prototypes = bool(update_prototypes)
        self.use_ema_for_dual = bool(use_ema_for_dual)
        self.chance_prior_for_dual = chance_prior_for_dual

        if efficacy_meter is not None:
            self.efficacy_meter = efficacy_meter.to(self.device)
        elif self.use_ema_for_dual:
            if ChanceCorrectedEfficacyMeter is None:
                raise ImportError("ChanceCorrectedEfficacyMeter is not available")
            num_classes = getattr(self.criterion, "num_classes", None)
            if num_classes is None:
                raise ValueError("criterion must expose num_classes to auto-create efficacy_meter")
            ignore_index = getattr(self.criterion, "ignore_index", 255 if task == "segmentation" else None)
            softmin_tau = getattr(getattr(self.criterion, "constraint", None), "softmin_tau", None)
            self.efficacy_meter = ChanceCorrectedEfficacyMeter(
                num_classes=int(num_classes),
                ignore_index=ignore_index,
                momentum=meter_momentum,
                softmin_tau=softmin_tau,
                min_class_count=min_class_count,
                min_valid_count=min_valid_count,
                warmup_updates=warmup_updates,
            ).to(self.device)
        else:
            self.efficacy_meter = None

    def forward_model(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        training: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        output_size = tuple(targets.shape[-2:]) if self.task == "segmentation" else None
        return self.model(
            images,
            target=targets,
            output_size=output_size,
            update_pred_prior=self.update_pred_prior and training,
            use_prior=self.use_prior,
            use_prototype=self.use_prototype,
            allow_target_prior=(training and self.allow_target_prior_train),
        )

    def _chance_prior_for_loss(self, outputs: Dict[str, Tensor]) -> Optional[Tensor]:
        prior_info = outputs.get("prior_info", {})
        if isinstance(prior_info, dict):
            return prior_info.get("chance_prior_for_loss", prior_info.get("batch_prior", prior_info.get("global_prior", None)))
        return None

    def _dual_inputs(
        self,
        outputs: Dict[str, Tensor],
        targets: Tensor,
        *,
        training: bool,
        chance_prior_loss: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        if not self.use_ema_for_dual or self.efficacy_meter is None:
            return {}

        if "evidence_prob" not in outputs:
            return {}

        # Default: let EMA meter use its EMA true prior for dual update instead
        # of mixing EMA confusion with current batch chance_prior. If explicitly
        # requested, use the same prior as the loss.
        chance_prior_dual = chance_prior_loss if self.chance_prior_for_dual == "loss" else None
        meter_out = self.efficacy_meter(
            probs=outputs["evidence_prob"].detach(),
            target=targets.detach(),
            chance_prior=chance_prior_dual,
            update=training,
        )

        dual_kwargs: Dict[str, Tensor] = {
            "dual_class_psi": meter_out["psi_for_dual"].detach(),
            "dual_update_mask": meter_out["dual_update_mask"].detach(),
        }
        if "map_psi_for_dual" in meter_out:
            dual_kwargs["dual_map_psi"] = meter_out["map_psi_for_dual"].detach()

        # Add scalar logs without interfering with loss computation.
        dual_kwargs["_meter_map_psi"] = meter_out["map_psi"].detach()
        dual_kwargs["_meter_map_psi_for_dual"] = meter_out["map_psi_for_dual"].detach()
        return dual_kwargs

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Tensor,
        *,
        training: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        chance_prior_loss = self._chance_prior_for_loss(outputs)
        dual_kwargs = self._dual_inputs(
            outputs,
            targets,
            training=training,
            chance_prior_loss=chance_prior_loss,
        )

        # Remove internal logging keys before passing into criterion.
        criterion_dual_kwargs = {k: v for k, v in dual_kwargs.items() if not k.startswith("_")}

        loss_dict = self.criterion(
            outputs=outputs,
            target=targets,
            chance_prior=chance_prior_loss,
            use_constraint=True,
            auto_update_mu=training,
            **criterion_dual_kwargs,
        )

        if "dual_class_psi" in criterion_dual_kwargs:
            loss_dict["psi_for_dual"] = criterion_dual_kwargs["dual_class_psi"].detach()
        if "dual_update_mask" in criterion_dual_kwargs:
            # Not scalar, so BaseTrainer logger will ignore it.
            loss_dict["dual_update_mask"] = criterion_dual_kwargs["dual_update_mask"].detach()
        if "dual_map_psi" in criterion_dual_kwargs:
            loss_dict["map_psi_for_dual"] = criterion_dual_kwargs["dual_map_psi"].detach()
        if "_meter_map_psi" in dual_kwargs:
            loss_dict["meter_map_psi"] = dual_kwargs["_meter_map_psi"].detach()
        if "_meter_map_psi_for_dual" in dual_kwargs:
            loss_dict["meter_map_psi_for_dual"] = dual_kwargs["_meter_map_psi_for_dual"].detach()
        return loss_dict

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

        # Prefer stable psi_for_dual if available; otherwise fall back to batch psi.
        psi = loss_dict.get("psi_for_dual", loss_dict.get("class_psi", loss_dict.get("psi", None)))
        if torch.is_tensor(psi):
            psi = psi.detach()

        self.model.update_prototypes(
            features=outputs["features"].detach(),
            target=targets,
            psi=psi,
        )

    def reset_efficacy_meter(self) -> None:
        if self.efficacy_meter is not None and hasattr(self.efficacy_meter, "reset"):
            self.efficacy_meter.reset()
