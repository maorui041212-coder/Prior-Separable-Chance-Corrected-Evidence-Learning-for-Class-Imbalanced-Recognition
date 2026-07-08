"""
Segmentation backbones for CCEL-Net.

This file provides a lightweight Mobile-UNETR style segmentation feature
extractor. It is intentionally written as a backbone, not a full CCEL-Net model:

    image -> MobileUNETRBackbone -> feature map [B, feature_dim, H/2, W/2]

CCELNet will attach its own EvidenceHead / PriorBranch on top of this feature
map. For ordinary CE baselines, this file also provides a small standalone
MobileUNETRSegmentationModel with a 1x1 segmentation head.

Recommended placement:
    ccel/models/segmentation_backbone.py

Example for CCEL-Net:
    from ccel.models.segmentation_backbone import build_segmentation_backbone
    from ccel.models.ccel_net import CCELNet

    backbone, feature_dim = build_segmentation_backbone(
        name="mobile_unetr_xxs",
        in_channels=3,
        feature_dim=128,
    )
    model = CCELNet(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=3,
        class_prior=train_set.get_class_priors_tensor(),
        task="segmentation",
        ignore_index=255,
    )
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic layers
# -----------------------------------------------------------------------------


def _make_divisible(v: int, divisor: int = 8) -> int:
    return int((v + divisor - 1) // divisor * divisor)


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("stride must be 1 or 2")

        hidden_dim = _make_divisible(int(round(in_channels * expand_ratio)))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if hidden_dim != in_channels:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNAct(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_dim,
                ),
                ConvBNAct(hidden_dim, out_channels, kernel_size=1, act=False),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_res_connect:
            out = out + x
        return out


class TransformerEncoderBlock(nn.Module):
    """Small transformer encoder block applied on flattened spatial tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class UpFuseBlock(nn.Module):
    """Upsample decoder feature and fuse with encoder skip feature."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels, kernel_size=3),
            InvertedResidual(out_channels, out_channels, stride=1, expand_ratio=2.0),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


# -----------------------------------------------------------------------------
# Mobile-UNETR style backbone
# -----------------------------------------------------------------------------


class MobileUNETRBackbone(nn.Module):
    """
    Lightweight Mobile-UNETR style feature extractor for segmentation.

    It combines:
        1. MobileNet-style inverted residual encoder;
        2. Transformer blocks at the bottleneck;
        3. UNet-style skip decoder.

    Output:
        feature map [B, feature_dim, H/2, W/2].

    The output is deliberately a feature map instead of class logits, because
    CCELNet attaches EvidenceHead(feature_dim, num_classes, task="segmentation")
    by itself. If your trainer expects logits directly, use
    MobileUNETRSegmentationModel below.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 128,
        widths: Tuple[int, int, int, int] = (32, 64, 128, 192),
        depths: Tuple[int, int, int, int] = (1, 2, 2, 2),
        transformer_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        expand_ratio: float = 4.0,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = widths
        d1, d2, d3, d4 = depths
        self.feature_dim = int(feature_dim)
        self.out_channels = int(feature_dim)
        self.return_dict = bool(return_dict)

        self.stem = ConvBNAct(in_channels, c1, kernel_size=3, stride=2)  # H/2
        self.enc1 = self._make_stage(c1, c1, depth=d1, stride=1, expand_ratio=expand_ratio)
        self.enc2 = self._make_stage(c1, c2, depth=d2, stride=2, expand_ratio=expand_ratio)  # H/4
        self.enc3 = self._make_stage(c2, c3, depth=d3, stride=2, expand_ratio=expand_ratio)  # H/8
        self.enc4 = self._make_stage(c3, c4, depth=d4, stride=2, expand_ratio=expand_ratio)  # H/16

        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    dim=c4,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(transformer_depth)
            ]
        )

        self.dec3 = UpFuseBlock(c4, c3, c3)  # H/8
        self.dec2 = UpFuseBlock(c3, c2, c2)  # H/4
        self.dec1 = UpFuseBlock(c2, c1, feature_dim)  # H/2
        self.out_proj = nn.Sequential(
            InvertedResidual(feature_dim, feature_dim, stride=1, expand_ratio=2.0),
            ConvBNAct(feature_dim, feature_dim, kernel_size=1),
        )

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int,
        expand_ratio: float,
    ) -> nn.Sequential:
        layers = [
            InvertedResidual(
                in_channels,
                out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
        ]
        for _ in range(max(0, depth - 1)):
            layers.append(
                InvertedResidual(
                    out_channels,
                    out_channels,
                    stride=1,
                    expand_ratio=expand_ratio,
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        x1 = self.enc1(self.stem(x))  # H/2
        x2 = self.enc2(x1)            # H/4
        x3 = self.enc3(x2)            # H/8
        x4 = self.enc4(x3)            # H/16
        x4 = self.transformer(x4)

        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        feat = self.out_proj(d1)
        return feat

    def forward(self, x: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        feat = self.forward_features(x)
        if self.return_dict:
            return {"features": feat}
        return feat


class MobileUNETRSegmentationModel(nn.Module):
    """Standalone segmentation model for ordinary CE/loss baselines."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        feature_dim: int = 128,
        widths: Tuple[int, int, int, int] = (32, 64, 128, 192),
        depths: Tuple[int, int, int, int] = (1, 2, 2, 2),
        transformer_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = MobileUNETRBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
            widths=widths,
            depths=depths,
            transformer_depth=transformer_depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        feat = self.backbone(x)
        logits = self.classifier(feat)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------


_PRESETS = {
    # Safer first choice for 128x128 patches / small GPU.
    "mobile_unetr_xxs": {
        "widths": (24, 48, 96, 160),
        "depths": (1, 1, 2, 2),
        "feature_dim": 96,
        "transformer_depth": 1,
        "num_heads": 4,
    },
    "mobile_unetr_xs": {
        "widths": (32, 64, 128, 192),
        "depths": (1, 2, 2, 2),
        "feature_dim": 128,
        "transformer_depth": 2,
        "num_heads": 4,
    },
    "mobile_unetr_s": {
        "widths": (40, 80, 160, 240),
        "depths": (1, 2, 3, 2),
        "feature_dim": 160,
        "transformer_depth": 2,
        "num_heads": 4,
    },
}


def build_segmentation_backbone(
    name: str = "mobile_unetr_xxs",
    *,
    in_channels: int = 3,
    feature_dim: Optional[int] = None,
    return_dict: bool = False,
    **kwargs,
) -> Tuple[nn.Module, int]:
    """
    Build a segmentation backbone for CCELNet.

    Returns:
        backbone, feature_dim

    Supported names:
        - mobile_unetr
        - mobileunetr
        - mobile_unetr_xxs
        - mobile_unetr_xs
        - mobile_unetr_s
    """
    key = name.lower().replace("-", "_")
    if key in {"mobileunetr", "mobile_unetr"}:
        key = "mobile_unetr_xxs"
    if key not in _PRESETS:
        raise ValueError(f"Unknown segmentation backbone: {name}. Available: {list(_PRESETS)}")

    cfg = dict(_PRESETS[key])
    cfg.update(kwargs)
    if feature_dim is not None:
        cfg["feature_dim"] = int(feature_dim)

    backbone = MobileUNETRBackbone(
        in_channels=in_channels,
        feature_dim=int(cfg["feature_dim"]),
        widths=tuple(cfg["widths"]),
        depths=tuple(cfg["depths"]),
        transformer_depth=int(cfg["transformer_depth"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg.get("dropout", 0.0)),
        return_dict=return_dict,
    )
    return backbone, int(cfg["feature_dim"])


def build_segmentation_model(
    name: str = "mobile_unetr_xxs",
    *,
    num_classes: int,
    in_channels: int = 3,
    feature_dim: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build a standalone segmentation model for CE / baseline losses."""
    key = name.lower().replace("-", "_")
    if key in {"mobileunetr", "mobile_unetr"}:
        key = "mobile_unetr_xxs"
    if key not in _PRESETS:
        raise ValueError(f"Unknown segmentation model: {name}. Available: {list(_PRESETS)}")

    cfg = dict(_PRESETS[key])
    cfg.update(kwargs)
    if feature_dim is not None:
        cfg["feature_dim"] = int(feature_dim)

    return MobileUNETRSegmentationModel(
        num_classes=num_classes,
        in_channels=in_channels,
        feature_dim=int(cfg["feature_dim"]),
        widths=tuple(cfg["widths"]),
        depths=tuple(cfg["depths"]),
        transformer_depth=int(cfg["transformer_depth"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )


__all__ = [
    "MobileUNETRBackbone",
    "MobileUNETRSegmentationModel",
    "build_segmentation_backbone",
    "build_segmentation_model",
]
