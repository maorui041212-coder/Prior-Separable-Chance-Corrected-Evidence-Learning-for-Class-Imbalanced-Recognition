"""Evidence branch heads for classification and segmentation."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


class EvidenceHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        task: Literal["classification", "segmentation"] = "segmentation",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.task = task
        self.num_classes = int(num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if task == "segmentation":
            self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        elif task == "classification":
            self.classifier = nn.Linear(in_channels, num_classes, bias=True)
        else:
            raise ValueError("task must be 'classification' or 'segmentation'")

    def forward(
        self,
        features: Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if self.task == "segmentation":
            if features.dim() != 4:
                raise ValueError(f"segmentation features must be [B,D,H,W], got {tuple(features.shape)}")
            x = self.dropout(features)
            logits = self.classifier(x)
            if output_size is not None and logits.shape[-2:] != tuple(output_size):
                logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            return logits

        # classification
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        elif features.dim() != 2:
            raise ValueError(f"classification features must be [B,D] or [B,D,H,W], got {tuple(features.shape)}")
        features = self.dropout(features)
        return self.classifier(features)
