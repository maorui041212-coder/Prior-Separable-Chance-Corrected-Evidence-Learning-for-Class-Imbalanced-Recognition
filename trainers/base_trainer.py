"""Base trainer utilities for CCEL-Net.

This module owns generic training-loop mechanics only:
    - device movement;
    - batch parsing;
    - mixed precision / gradient clipping;
    - epoch-level train / validation loops;
    - checkpoint save/load;
    - scalar logging helpers;
    - optional metric hooks and best-checkpoint bookkeeping.

Task-specific forward/loss/metric behavior belongs to segmentation_trainer.py,
classification_trainer.py, prior_warmup_trainer.py, or primal_dual_trainer.py.

Important for CCEL-Net
----------------------
Stage-3 primal-dual training stores non-model state outside the model, e.g.
PrimalDualEfficacyLoss.mu and ChanceCorrectedEfficacyMeter EMA buffers.
Therefore checkpointing must save optional ``criterion`` and ``efficacy_meter``
attributes when subclasses define them.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class TrainerState:
    """Lightweight state shared by all trainers."""

    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    history: list[Dict[str, float]] = field(default_factory=list)


class AverageMeter:
    """Accumulate scalar averages."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


class BaseTrainer:
    """Base class for CCEL-Net trainers.

    Subclasses should override:
        - forward_model(...)
        - compute_loss(...)

    Optional hooks:
        - compute_metrics(...)
        - after_optimizer_step(...)

    Optional attributes saved in checkpoints if present:
        - criterion
        - efficacy_meter
    """

    input_keys = ("image", "images", "img", "x", "input", "inputs")
    target_keys = ("target", "targets", "mask", "masks", "label", "labels", "y")

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        device: Optional[torch.device | str] = None,
        scheduler: Optional[Any] = None,
        amp: bool = False,
        scaler: Optional[Any] = None,
        grad_clip_norm: Optional[float] = None,
        scheduler_step_on: str = "epoch",
        log_interval: int = 50,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device) if device is not None else self._infer_device(model)
        self.scheduler = scheduler
        self.amp = bool(amp)
        self.scaler = scaler
        self.grad_clip_norm = grad_clip_norm
        if scheduler_step_on not in {"epoch", "step", "none"}:
            raise ValueError("scheduler_step_on must be 'epoch', 'step', or 'none'")
        self.scheduler_step_on = scheduler_step_on
        self.log_interval = int(log_interval)
        self.state = TrainerState()
        self.model.to(self.device)

    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def move_to_device(self, obj: Any) -> Any:
        """Recursively move tensors in a batch to trainer device."""
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, Mapping):
            return {k: self.move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self.move_to_device(v) for v in obj)
        if isinstance(obj, list):
            return [self.move_to_device(v) for v in obj]
        return obj

    def unpack_batch(self, batch: Any) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Return images, targets, and metadata.

        Supported forms:
            - dict with image/images/x and target/mask/label keys;
            - tuple/list: (images, targets) or (images, targets, metadata).
        """
        metadata: Dict[str, Any] = {}
        if isinstance(batch, Mapping):
            images = None
            targets = None
            for k in self.input_keys:
                if k in batch:
                    images = batch[k]
                    break
            for k in self.target_keys:
                if k in batch:
                    targets = batch[k]
                    break
            if images is None or targets is None:
                raise KeyError(
                    f"Cannot find image/target keys in batch. Available keys: {list(batch.keys())}"
                )
            metadata = {k: v for k, v in batch.items() if k not in set(self.input_keys + self.target_keys)}
            return images, targets, metadata

        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError("Tuple/list batch must contain at least (images, targets)")
            images, targets = batch[0], batch[1]
            if len(batch) >= 3 and isinstance(batch[2], Mapping):
                metadata = dict(batch[2])
            return images, targets, metadata

        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    def autocast_context(self):
        """Return an autocast context only when AMP is enabled on CUDA."""
        if self.amp and self.device.type == "cuda":
            # torch.amp.autocast is the preferred API in recent PyTorch.
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast(device_type="cuda")
            return torch.cuda.amp.autocast()
        return nullcontext()

    @staticmethod
    def _ensure_scalar_loss(loss: Any) -> Tensor:
        """Validate that compute_loss returned a scalar tensor loss."""
        if not torch.is_tensor(loss):
            raise TypeError("loss_dict['loss'] must be a torch.Tensor")
        if loss.numel() != 1:
            raise ValueError(
                "loss_dict['loss'] must be a scalar tensor. "
                f"Got shape {tuple(loss.shape)} with numel={loss.numel()}."
            )
        return loss

    def backward_and_step(self, loss: Tensor) -> None:
        loss = self._ensure_scalar_loss(loss)
        if self.optimizer is None:
            raise RuntimeError("optimizer is required for training")
        self.optimizer.zero_grad(set_to_none=True)

        if self.amp and self.scaler is not None and self.device.type == "cuda":
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        if self.scheduler is not None and self.scheduler_step_on == "step":
            self.scheduler.step()

    def forward_model(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        training: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Tensor,
        *,
        training: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def compute_metrics(
        self,
        outputs: Dict[str, Tensor],
        targets: Tensor,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Optional validation metric hook.

        BaseTrainer only provides the hook. Task trainers should override this
        to report mIoU, minority IoU/F1, OA, balanced accuracy, macro-F1, ECE,
        efficacy, etc.
        """
        return {}

    def after_optimizer_step(
        self,
        outputs: Dict[str, Tensor],
        loss_dict: Dict[str, Tensor],
        targets: Tensor,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Hook for prototype EMA or custom state updates."""
        return None

    @staticmethod
    def _batch_size(images: Tensor) -> int:
        return int(images.size(0)) if torch.is_tensor(images) and images.dim() > 0 else 1

    @staticmethod
    def _scalar_logs(loss_dict: Mapping[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v) and v.numel() == 1:
                out[k] = float(v.detach().cpu())
            elif isinstance(v, (int, float)):
                out[k] = float(v)
        return out

    @staticmethod
    def _update_meters(meters: MutableMapping[str, AverageMeter], logs: Mapping[str, float], n: int) -> None:
        for k, v in logs.items():
            meters.setdefault(k, AverageMeter()).update(v, n=n)

    @staticmethod
    def _meters_to_dict(meters: Mapping[str, AverageMeter], prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}{k}": meter.avg for k, meter in meters.items()}

    @staticmethod
    def _call_ce(
        logits: Tensor,
        targets: Tensor,
        *,
        ignore_index: Optional[int] = None,
        weight: Optional[Tensor] = None,
    ) -> Tensor:
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=weight,
            ignore_index=ignore_index if ignore_index is not None else -100,
        )

    def train_one_epoch(self, loader: Iterable[Any], *, epoch: Optional[int] = None) -> Dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("optimizer is required for train_one_epoch")
        if epoch is not None:
            self.state.epoch = int(epoch)

        self.model.train()
        meters: Dict[str, AverageMeter] = {}

        for step, batch in enumerate(loader):
            batch = self.move_to_device(batch)
            images, targets, metadata = self.unpack_batch(batch)

            with self.autocast_context():
                outputs = self.forward_model(images, targets, training=True, metadata=metadata)
                loss_dict = self.compute_loss(outputs, targets, training=True, metadata=metadata)
                if "loss" not in loss_dict:
                    raise KeyError("compute_loss must return a dict containing key 'loss'")
                loss = self._ensure_scalar_loss(loss_dict["loss"])

            self.backward_and_step(loss)
            self.after_optimizer_step(outputs, loss_dict, targets, metadata=metadata)

            logs = self._scalar_logs(loss_dict)
            self._update_meters(meters, logs, n=self._batch_size(images))
            self.state.global_step += 1

        if self.scheduler is not None and self.scheduler_step_on == "epoch":
            self.scheduler.step()

        logs = self._meters_to_dict(meters, prefix="train/")
        logs["epoch"] = float(self.state.epoch)
        self.state.history.append(logs)
        return logs

    @torch.no_grad()
    def validate(self, loader: Iterable[Any]) -> Dict[str, float]:
        self.model.eval()
        meters: Dict[str, AverageMeter] = {}

        for batch in loader:
            batch = self.move_to_device(batch)
            images, targets, metadata = self.unpack_batch(batch)
            with self.autocast_context():
                outputs = self.forward_model(images, targets, training=False, metadata=metadata)
                loss_dict = self.compute_loss(outputs, targets, training=False, metadata=metadata)
                if "loss" in loss_dict:
                    self._ensure_scalar_loss(loss_dict["loss"])

            logs = self._scalar_logs(loss_dict)
            metric_logs = self.compute_metrics(outputs, targets, metadata=metadata)
            logs.update(metric_logs)
            self._update_meters(meters, logs, n=self._batch_size(images))

        return self._meters_to_dict(meters, prefix="val/")

    @staticmethod
    def _is_better(metric: float, best: Optional[float], mode: str) -> bool:
        if mode not in {"max", "min"}:
            raise ValueError("monitor_mode must be 'max' or 'min'")
        if best is None:
            return True
        return metric > best if mode == "max" else metric < best

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        val_loader: Optional[Iterable[Any]] = None,
        epochs: int,
        monitor: Optional[str] = None,
        monitor_mode: str = "max",
        save_dir: Optional[str | Path] = None,
        save_last: bool = False,
        save_best: bool = True,
    ) -> list[Dict[str, float]]:
        """Run training for ``epochs``.

        Args:
            monitor:
                Metric key used for best checkpoint, e.g. ``val/C1_IoU`` or
                ``val/minority_IoU``. If the exact key is absent and it does not
                start with ``val/``, ``val/{monitor}`` is also tried.
            monitor_mode:
                ``max`` for metrics like IoU/accuracy, ``min`` for loss.
            save_dir:
                If given, save ``last.pt`` and/or ``best.pt``.
        """
        history: list[Dict[str, float]] = []
        save_path: Optional[Path] = Path(save_dir) if save_dir is not None else None
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)

        for ep in range(int(epochs)):
            self.state.epoch = ep
            train_logs = self.train_one_epoch(train_loader, epoch=ep)
            logs = dict(train_logs)
            if val_loader is not None:
                logs.update(self.validate(val_loader))

            if monitor is not None:
                key = monitor
                if key not in logs and not key.startswith("val/"):
                    key = f"val/{monitor}"
                if key not in logs:
                    raise KeyError(
                        f"Monitor key '{monitor}' not found in logs. Available keys: {sorted(logs.keys())}"
                    )
                metric = float(logs[key])
                if self._is_better(metric, self.state.best_metric, monitor_mode):
                    self.state.best_metric = metric
                    if save_path is not None and save_best:
                        self.save_checkpoint(save_path / "best.pt")

            if save_path is not None and save_last:
                self.save_checkpoint(save_path / "last.pt")

            history.append(logs)
        return history

    @staticmethod
    def _state_dict_or_none(obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None or not hasattr(obj, "state_dict"):
            return None
        return obj.state_dict()

    def checkpoint_state(self) -> Dict[str, Any]:
        criterion = getattr(self, "criterion", None)
        efficacy_meter = getattr(self, "efficacy_meter", None)
        return {
            "model": self.model.state_dict(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "criterion": self._state_dict_or_none(criterion),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "efficacy_meter": self._state_dict_or_none(efficacy_meter),
            "state": asdict(self.state),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), path)

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=strict)

        if self.optimizer is not None and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        criterion = getattr(self, "criterion", None)
        if criterion is not None and ckpt.get("criterion") is not None:
            criterion.load_state_dict(ckpt["criterion"], strict=strict)

        if self.scaler is not None and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

        efficacy_meter = getattr(self, "efficacy_meter", None)
        if efficacy_meter is not None and ckpt.get("efficacy_meter") is not None:
            efficacy_meter.load_state_dict(ckpt["efficacy_meter"], strict=strict)

        state_obj = ckpt.get("state")
        if isinstance(state_obj, TrainerState):
            self.state = state_obj
        elif isinstance(state_obj, Mapping):
            self.state = TrainerState(**dict(state_obj))

        return ckpt
