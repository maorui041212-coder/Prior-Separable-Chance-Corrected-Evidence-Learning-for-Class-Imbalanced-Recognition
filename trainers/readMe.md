# CCEL-Net trainer responsibilities

This patch implements `ccel/trainers/`.

## File responsibilities

### `base_trainer.py`
Generic loop mechanics only:
- device movement;
- batch parsing for dict/tuple dataloaders;
- AMP / gradient clipping;
- train / validation loops;
- checkpoint save/load;
- scalar logging.

It does not know CCEL formulas.

### `prior_warmup_trainer.py`
Stage 1: strict ordinary CE warm-up.

It forwards CCELNet with:
- `use_prior=False`;
- `use_prototype=False`;
- `allow_target_prior=False`.

Therefore the warm-up objective is:

```text
CE(e_linear, y)
```

not `CE(b + e)` and not `CE(e + prototype)`.

It may still update:
- prediction-prior EMA from pure evidence predictions;
- prototype memory after optimizer step, only for initialization.

### `segmentation_trainer.py`
General segmentation trainer for CE baseline or Stage-2 evidence decoupling.
Typical Stage-2 call:

```python
SegmentationTrainer(
    model,
    optimizer,
    use_prior=True,
    use_prototype=False,
    allow_target_prior_train=True,
)
```

Validation/test always set `allow_target_prior=False` internally.

### `classification_trainer.py`
Classification counterpart of `segmentation_trainer.py`.

### `primal_dual_trainer.py`
Stage 3: primal-dual efficacy training.

It wires together:
- CCELNet forward with `z=b+e`;
- `PrimalDualEfficacyLoss` for `CE + mu * violation`;
- optional `ChanceCorrectedEfficacyMeter` for EMA-smoothed dual updates;
- explicit prototype memory update after `optimizer.step()`.

It does not reimplement efficacy formulas or violation logic.

## Recommended three-stage usage

```python
# Stage 1
warmup = PriorWarmupTrainer(model, optimizer, task="segmentation")
warmup.fit(train_loader, val_loader=val_loader, epochs=warmup_epochs)

# Stage 2
seg_trainer = SegmentationTrainer(
    model,
    optimizer,
    use_prior=True,
    use_prototype=False,
)
seg_trainer.fit(train_loader, val_loader=val_loader, epochs=decouple_epochs)

# Stage 3
pd_trainer = PrimalDualTrainer(
    model,
    optimizer,
    criterion=primal_dual_criterion,
    task="segmentation",
    use_prior=True,
    use_prototype=True,
    use_ema_for_dual=True,
)
pd_trainer.fit(train_loader, val_loader=val_loader, epochs=pd_epochs)
```
