from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = 255, weight=None) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        target.long(),
        weight=weight,
        ignore_index=ignore_index if ignore_index is not None else -100,
    )
