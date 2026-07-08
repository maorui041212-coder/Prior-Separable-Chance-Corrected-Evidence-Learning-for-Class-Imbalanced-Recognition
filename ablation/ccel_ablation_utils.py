"""
Utilities for CCEL-Net ablation experiments.

Put this file under:
    /data2/mr/MICE/CCEL_NET/ccel/ablation/ccel_ablation_utils.py
or directly import it from your scripts folder.

This file does NOT build datasets/backbones. It only controls the ablation logic:
    full
    no_prior
    no_constraint
    no_proto
    fixed_lambda
    freq_sampling
    miceloss_only
    logit_adjustment
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import WeightedRandomSampler

try:
    from ccel.models.ccel_net import CCELNet
except Exception:  # support local placement during quick testing
    CCELNet = None  # type: ignore

from ccel.losses.primal_dual_loss import PrimalDualEfficacyLoss
from ccel.losses.logit_adjustment_loss import LogitAdjustedCrossEntropyLoss


ABLATION_CHOICES = [
    "full",
    "no_prior",
    "no_constraint",
    "no_proto",
    "fixed_lambda",
    "freq_sampling",
    "miceloss_only",
    "logit_adjustment",
]


@dataclass
class AblationSwitches:
    """All switches needed by one train/eval step."""

    ablation: str

    # model construction
    build_ccel: bool
    build_prototype_memory: bool

    # forward-time switches
    use_prior: bool
    use_prototype: bool
    allow_target_prior: bool

    # loss-time switches
    use_constraint: bool
    auto_update_mu: bool
    use_fixed_lambda: bool
    use_miceloss_only: bool
    use_logit_adjustment_only: bool

    # sampler switches
    use_efficacy_guided_sampling: bool
    use_frequency_sampling: bool

    # state update switches
    update_prototype_memory: bool


def add_ablation_args(parser):
    """Add common ablation arguments to your existing argparse parser."""
    parser.add_argument("--ablation", type=str, default="full", choices=ABLATION_CHOICES)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--prototype-start-epoch", type=int, default=10)
    parser.add_argument("--fixed-lambda", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--eta-mu", type=float, default=0.01)
    parser.add_argument("--mu-max", type=float, default=10.0)
    parser.add_argument("--la-tau", type=float, default=1.0)
    parser.add_argument("--la-direction", type=str, default="add", choices=["add", "subtract"])
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--eval-evidence-only", action="store_true")
    parser.add_argument("--no-target-prior-in-train", action="store_true")
    return parser


def resolve_ablation_switches(args: Any, epoch: int, training: bool = True) -> AblationSwitches:
    """
    Resolve model/loss/sampler switches for current epoch.

    Important:
    - Stage-1 warm-up is strict CE: no prior, no prototype, no constraint.
    - miceloss_only and logit_adjustment are ordinary-baseline modes.
    - no_prior removes prior logits from final prediction.
    - no_proto removes prototype memory contribution.
    - no_constraint keeps CCEL structure but disables efficacy constraint and mu update.
    - fixed_lambda keeps efficacy violation but freezes mu to fixed_lambda.
    """
    ablation = str(args.ablation)
    if ablation not in ABLATION_CHOICES:
        raise ValueError(f"Unknown ablation={ablation}. Choices: {ABLATION_CHOICES}")

    in_warmup = training and epoch < int(args.warmup_epochs)
    proto_active = training and epoch >= int(args.prototype_start_epoch)

    is_baseline = ablation in {"miceloss_only", "logit_adjustment"}

    build_ccel = not is_baseline
    build_proto = build_ccel and ablation != "no_proto"

    if in_warmup:
        use_prior = False
        use_proto = False
        use_constraint = False
        auto_update_mu = False
        update_proto = False
    else:
        use_prior = build_ccel and ablation != "no_prior"
        use_proto = build_proto and proto_active
        use_constraint = build_ccel and ablation != "no_constraint"
        auto_update_mu = use_constraint and ablation != "fixed_lambda"
        update_proto = use_proto

    # Evaluation defaults. Do not use target-derived prior during eval.
    if not training:
        if bool(getattr(args, "eval_evidence_only", False)):
            use_prior = False
            use_proto = False
        else:
            use_prior = build_ccel and ablation != "no_prior"
            use_proto = build_proto
        use_constraint = False
        auto_update_mu = False
        update_proto = False

    allow_target_prior = bool(training and not bool(getattr(args, "no_target_prior_in_train", False)))

    return AblationSwitches(
        ablation=ablation,
        build_ccel=build_ccel,
        build_prototype_memory=build_proto,
        use_prior=use_prior,
        use_prototype=use_proto,
        allow_target_prior=allow_target_prior,
        use_constraint=use_constraint,
        auto_update_mu=auto_update_mu,
        use_fixed_lambda=ablation == "fixed_lambda" and not in_warmup,
        use_miceloss_only=ablation == "miceloss_only",
        use_logit_adjustment_only=ablation == "logit_adjustment",
        use_efficacy_guided_sampling=ablation not in {"freq_sampling", "miceloss_only", "logit_adjustment"},
        use_frequency_sampling=ablation == "freq_sampling",
        update_prototype_memory=update_proto,
    )


def build_ccel_ablation_model(
    *,
    backbone: nn.Module,
    feature_dim: int,
    num_classes: int,
    class_prior: Sequence[float] | Tensor,
    task: str,
    args: Any,
    feature_key: Optional[str] = None,
    prior_kwargs: Optional[Dict[str, Any]] = None,
    resize_logits_to_target: bool = True,
) -> nn.Module:
    """
    Build CCEL-Net for all CCEL variants.

    For miceloss_only/logit_adjustment, do NOT call this. Use your ordinary
    baseline model instead, or use CCELNet with use_prior=False/use_proto=False
    only as a temporary debugging substitute.
    """
    if CCELNet is None:
        raise ImportError("Cannot import ccel.models.ccel_net.CCELNet. Check your PYTHONPATH.")

    switches = resolve_ablation_switches(args, epoch=999999, training=False)
    return CCELNet(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        class_prior=class_prior,
        task=task,  # "classification" or "segmentation"
        feature_key=feature_key,
        use_prior_branch=True,  # keep object available; ablation controls use_prior at forward
        use_prototype_memory=switches.build_prototype_memory,
        prototype_momentum=float(getattr(args, "prototype_momentum", 0.99)),
        prototype_lambda_p=float(getattr(args, "prototype_lambda_p", 0.1)),
        prototype_eta=float(getattr(args, "prototype_eta", 0.1)),
        learnable_eta=bool(getattr(args, "learnable_eta", True)),
        max_prototype_eta=getattr(args, "max_prototype_eta", 5.0),
        ignore_index=getattr(args, "ignore_index", 255),
        prior_kwargs=prior_kwargs,
        resize_logits_to_target=resize_logits_to_target,
    )


def build_primal_dual_loss_for_ablation(
    *,
    num_classes: int,
    target_classes: Optional[Iterable[int]],
    args: Any,
    task: str,
    ce_weight: Optional[Tensor] = None,
) -> PrimalDualEfficacyLoss:
    """Build primal-dual loss. It can also be used for no_constraint/fixed_lambda."""
    use_map_constraint = bool(task == "segmentation" and getattr(args, "use_map_constraint", False))
    map_rho = getattr(args, "map_rho", None)
    return PrimalDualEfficacyLoss(
        num_classes=num_classes,
        rho=getattr(args, "rho", 0.1),
        target_classes=target_classes,
        eta_mu=getattr(args, "eta_mu", 0.01),
        mu_max=getattr(args, "mu_max", 10.0),
        ignore_index=getattr(args, "ignore_index", 255),
        ce_weight=ce_weight,
        use_map_constraint=use_map_constraint,
        map_rho=map_rho,
        auto_update_mu=True,
    )


def build_logit_adjustment_loss_for_ablation(
    *,
    num_classes: int,
    class_prior: Sequence[float] | Tensor,
    args: Any,
) -> LogitAdjustedCrossEntropyLoss:
    return LogitAdjustedCrossEntropyLoss(
        num_classes=num_classes,
        class_prior=class_prior,
        tau=float(getattr(args, "la_tau", 1.0)),
        direction=str(getattr(args, "la_direction", "add")),
        ignore_index=getattr(args, "ignore_index", 255),
        max_abs_adjustment=None,  # fair standard Logit Adjustment
    )


@torch.no_grad()
def set_fixed_lambda_mu(
    loss_fn: PrimalDualEfficacyLoss,
    *,
    fixed_lambda: float,
    target_classes: Optional[Iterable[int]],
) -> None:
    """
    Freeze dual variables to a constant lambda.

    Call once before training and after loading checkpoints if ablation=fixed_lambda.
    """
    loss_fn.mu.zero_()
    if target_classes is None:
        loss_fn.mu.fill_(float(fixed_lambda))
    else:
        for c in target_classes:
            loss_fn.mu[int(c)] = float(fixed_lambda)


def forward_with_ablation(
    model: nn.Module,
    images: Tensor,
    targets: Optional[Tensor],
    *,
    switches: AblationSwitches,
    output_size: Optional[tuple[int, int]] = None,
) -> Dict[str, Tensor]:
    """Forward pass for CCEL variants."""
    return model(
        images,
        target=targets,
        output_size=output_size,
        update_pred_prior=True,
        use_prior=switches.use_prior,
        use_prototype=switches.use_prototype,
        allow_target_prior=switches.allow_target_prior,
    )


def compute_ablation_loss(
    *,
    switches: AblationSwitches,
    outputs: Dict[str, Tensor] | Tensor,
    targets: Tensor,
    primal_dual_loss: Optional[PrimalDualEfficacyLoss] = None,
    miceloss: Optional[nn.Module] = None,
    logit_adjustment_loss: Optional[LogitAdjustedCrossEntropyLoss] = None,
    fixed_lambda: Optional[float] = None,
    target_classes: Optional[Iterable[int]] = None,
) -> Dict[str, Tensor]:
    """
    Compute loss for one batch.

    outputs:
        CCEL variants: dict from CCELNet.forward
        miceloss/logit_adjustment baseline: raw logits tensor or dict containing evidence_logits_linear/logits
    """
    if switches.use_miceloss_only:
        if miceloss is None:
            raise ValueError("miceloss must be provided when ablation=miceloss_only")
        logits = _extract_baseline_logits(outputs)
        loss = miceloss(logits, targets)
        return {"loss": loss, "ce_loss": loss.detach(), "constraint_loss": torch.zeros_like(loss.detach())}

    if switches.use_logit_adjustment_only:
        if logit_adjustment_loss is None:
            raise ValueError("logit_adjustment_loss must be provided when ablation=logit_adjustment")
        logits = _extract_baseline_logits(outputs)
        loss = logit_adjustment_loss(logits, targets)
        return {"loss": loss, "ce_loss": loss.detach(), "constraint_loss": torch.zeros_like(loss.detach())}

    if primal_dual_loss is None:
        raise ValueError("primal_dual_loss must be provided for CCEL variants")
    if not isinstance(outputs, dict):
        raise TypeError("CCEL variants require outputs dict from CCELNet.forward")

    if switches.use_fixed_lambda:
        if fixed_lambda is None:
            raise ValueError("fixed_lambda must be provided when ablation=fixed_lambda")
        set_fixed_lambda_mu(
            primal_dual_loss,
            fixed_lambda=float(fixed_lambda),
            target_classes=target_classes,
        )

    chance_prior = outputs.get("prior_info", {}).get("chance_prior_for_loss", None)
    loss_out = primal_dual_loss(
        outputs,
        targets,
        chance_prior=chance_prior,
        use_constraint=switches.use_constraint,
        auto_update_mu=switches.auto_update_mu,
    )
    return loss_out


def _extract_baseline_logits(outputs: Dict[str, Tensor] | Tensor) -> Tensor:
    if torch.is_tensor(outputs):
        return outputs
    for key in ["evidence_logits_linear", "logits", "out"]:
        if key in outputs:
            return outputs[key]
    raise KeyError("Cannot extract logits from outputs. Expected tensor or key logits/evidence_logits_linear/out.")


@torch.no_grad()
def update_prototype_after_step(
    *,
    model: nn.Module,
    outputs: Dict[str, Tensor],
    targets: Tensor,
    switches: AblationSwitches,
    loss_out: Optional[Dict[str, Tensor]] = None,
) -> None:
    """Update prototype memory after optimizer.step()."""
    if not switches.update_prototype_memory:
        return
    if not hasattr(model, "update_prototypes"):
        return
    psi = None
    if loss_out is not None:
        psi = loss_out.get("class_psi", loss_out.get("psi", None))
        if psi is not None:
            psi = psi.detach()
    model.update_prototypes(features=outputs["features"].detach(), target=targets, psi=psi)


# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------

def make_frequency_sampler_for_classification(
    labels: Sequence[int] | Tensor,
    num_classes: int,
    class_counts: Optional[Sequence[int] | Tensor] = None,
    eps: float = 1.0,
) -> WeightedRandomSampler:
    """
    Class-frequency sampler for classification.

    labels: one label per training image/sample.
    class_counts: training class counts. If None, inferred from labels.
    """
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    if class_counts is None:
        counts = torch.bincount(labels_t.clamp_min(0), minlength=num_classes).float()
    else:
        counts = torch.as_tensor(class_counts, dtype=torch.float32).reshape(-1)
        if counts.numel() != num_classes:
            raise ValueError(f"class_counts must have {num_classes} elements")

    class_weights = 1.0 / counts.clamp_min(float(eps))
    sample_weights = class_weights[labels_t].double()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def make_frequency_sampler_from_dataset(
    dataset: Any,
    num_classes: int,
    class_counts: Optional[Sequence[int] | Tensor] = None,
    label_getter: Optional[Callable[[Any, int], int]] = None,
) -> WeightedRandomSampler:
    """
    Frequency sampler when labels are stored in different dataset formats.

    Default tries:
        dataset.samples[i][1]
        dataset.targets[i]
        dataset.labels[i]
    """
    labels = []
    for i in range(len(dataset)):
        if label_getter is not None:
            y = label_getter(dataset, i)
        elif hasattr(dataset, "samples"):
            y = dataset.samples[i][1]
        elif hasattr(dataset, "targets"):
            y = dataset.targets[i]
        elif hasattr(dataset, "labels"):
            y = dataset.labels[i]
        else:
            raise AttributeError("Cannot infer labels. Provide label_getter(dataset, i).")
        labels.append(int(y))
    return make_frequency_sampler_for_classification(labels, num_classes, class_counts)


def update_class_sampling_weights_from_psi(
    class_psi: Tensor,
    *,
    alpha: float = 1.0,
    min_weight: float = 1e-3,
    max_weight: float = 100.0,
) -> Tensor:
    """
    Efficacy-guided class weight: weight_c ∝ (1 - psi_c)^alpha.

    Use this to update your custom sampler between epochs.
    """
    psi = class_psi.detach().float().clamp(min=-1.0, max=1.0)
    gap = (1.0 - psi).clamp_min(0.0)
    w = gap.pow(float(alpha)).clamp(min=float(min_weight), max=float(max_weight))
    w = w / w.mean().clamp_min(1e-12)
    return w


def get_logits_for_metrics(outputs: Dict[str, Tensor] | Tensor, *, evidence_only: bool = False) -> Tensor:
    """Choose logits for validation/test metric accumulation."""
    if torch.is_tensor(outputs):
        return outputs
    if evidence_only:
        return outputs["evidence_logits"]
    return outputs["logits"]
