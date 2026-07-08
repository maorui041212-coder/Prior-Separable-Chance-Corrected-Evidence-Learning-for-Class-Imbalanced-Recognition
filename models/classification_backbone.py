"""Classification backbones for CCEL-Net.

All backbones in this file are feature extractors only.
They follow the project convention:

    backbone(x) -> {"features": h}

For image classification, h is a 2D tensor with shape [B, D].
CCELNet or a simple Linear classifier can then consume h.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import nn

Tensor = torch.Tensor


class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor.

    Output:
        {"features": h}, where h is [B, 2048].

    Args:
        pretrained:
            Use torchvision ImageNet weights.
        freeze_stem:
            Freeze conv1/bn1.
        cifar_stem:
            Replace the ImageNet 7x7 stride-2 stem with a CIFAR-friendly
            3x3 stride-1 stem and remove maxpool.
    """

    def __init__(
        self,
        pretrained: bool = False,
        freeze_stem: bool = False,
        cifar_stem: bool = True,
    ) -> None:
        super().__init__()

        try:
            import torchvision.models as models
        except Exception as exc:  # pragma: no cover
            raise ImportError("torchvision is required for ResNet50Backbone.") from exc

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

        if cifar_stem:
            model.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            model.maxpool = nn.Identity()

        self.feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()

        self.model = model
        self.cifar_stem = bool(cifar_stem)

        if freeze_stem:
            for name, p in self.model.named_parameters():
                if name.startswith("conv1") or name.startswith("bn1"):
                    p.requires_grad = False

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.model(x)
        if h.dim() != 2:
            h = h.flatten(1)
        return {"features": h}


class MambaVisionBackbone(nn.Module):
    """MambaVision feature extractor through Hugging Face AutoModel.

    Recommended model names:
        nvidia/MambaVision-T-1K
        nvidia/MambaVision-S-1K
    """

    def __init__(
        self,
        model_name: str = "nvidia/MambaVision-T-1K",
        pretrained: bool = True,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for MambaVisionBackbone. "
                "Install with: pip install transformers"
            ) from exc

        if not pretrained:
            raise ValueError(
                "MambaVisionBackbone expects pretrained=True through Hugging Face/local directory."
            )

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.feature_dim = self._infer_feature_dim()

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):  # keep frozen backbone in eval mode
        super().train(mode)
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self

    def _infer_feature_dim(self) -> int:
        cfg = getattr(self.model, "config", None)
        for key in ("hidden_size", "num_features", "feature_dim", "embed_dim"):
            if cfg is not None and hasattr(cfg, key):
                value = getattr(cfg, key)
                if isinstance(value, int):
                    return int(value)

        name = self.model_name.lower()
        if "mambavision-t" in name:
            return 640
        if "mambavision-s" in name:
            return 768
        if "mambavision-b" in name:
            return 1024
        raise ValueError("Could not infer MambaVision feature_dim.")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = self.model(x)

        if isinstance(out, tuple):
            h = out[0]
        elif isinstance(out, dict):
            if "pooler_output" in out:
                h = out["pooler_output"]
            elif "last_hidden_state" in out:
                h = out["last_hidden_state"]
                if h.dim() > 2:
                    h = h.mean(dim=1)
            elif "features" in out:
                h = out["features"]
            else:
                raise KeyError(f"Unknown MambaVision output keys: {list(out.keys())}")
        else:
            h = out

        if not torch.is_tensor(h):
            raise TypeError("MambaVision backbone output feature is not a tensor")
        if h.dim() > 2:
            h = h.flatten(1)
        return {"features": h}


class DINOv3ViTS16Backbone(nn.Module):
    """DINOv3 ViT-S/16 feature extractor through Hugging Face Transformers.

    Default checkpoint:
        facebook/dinov3-vits16-pretrain-lvd1689m

    Output:
        {"features": h}, where h is [B, 384].

    Notes:
        1. Inputs must already be resized and normalized by the dataloader.
           For LVD-1689M weights, use ImageNet mean/std.
        2. Use --local_files_only with a local checkpoint directory when the
           server cannot access Hugging Face.
        3. For small long-tailed datasets, frozen DINOv3 + trainable head is a
           stable first run. Fine-tuning is available by setting freeze=False.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        pretrained: bool = True,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        freeze: bool = False,
        use_pooler_output: bool = True,
    ) -> None:
        super().__init__()

        if not pretrained:
            raise ValueError(
                "DINOv3ViTS16Backbone is intended to use pretrained DINOv3 weights. "
                "For training a ViT from scratch, use timm/torchvision ViT instead."
            )

        try:
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "transformers>=4.56.0 is required for DINOv3 Hugging Face loading. "
                "Install with: pip install -U transformers"
            ) from exc

        self.model_name_or_path = model_name_or_path
        self.use_pooler_output = bool(use_pooler_output)

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

        cfg = getattr(self.model, "config", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(cfg, "embed_dim", None)
        if hidden_size is None:
            hidden_size = 384  # DINOv3 ViT-S/16 default
        self.feature_dim = int(hidden_size)

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):  # keep frozen backbone in eval mode
        super().train(mode)
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = self.model(pixel_values=x)

        h: Tensor
        if self.use_pooler_output and hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            # CLS token. DINOv3 ViT output is [B, 1 + registers + patches, D].
            h = out.last_hidden_state[:, 0, :]
        elif isinstance(out, dict):
            if self.use_pooler_output and out.get("pooler_output", None) is not None:
                h = out["pooler_output"]
            elif out.get("last_hidden_state", None) is not None:
                h = out["last_hidden_state"][:, 0, :]
            else:
                raise KeyError(f"Unknown DINOv3 output keys: {list(out.keys())}")
        elif isinstance(out, tuple):
            first = out[0]
            h = first[:, 0, :] if first.dim() == 3 else first
        else:
            h = out

        if not torch.is_tensor(h):
            raise TypeError("DINOv3 output feature is not a tensor")
        if h.dim() > 2:
            h = h.flatten(1)
        return {"features": h}


class OfficialDINOv3ViTS16Backbone(nn.Module):
    """DINOv3 ViT-S/16 feature extractor through official Meta repo + .pth.

    Loading style follows the official DINOv3 PyTorch Hub interface:
        torch.hub.load(REPO_DIR, "dinov3_vits16", source="local", weights=WEIGHTS)

    Output:
        {"features": h}, where h is [B, 384].

    This class is for the Kaggle/Meta .pth file, e.g.
        dinov3_vits16_pretrain_lvd1689m-08c60483.pth
    It does not need Hugging Face Transformers.
    """

    def __init__(
        self,
        repo_dir: str,
        weights: str,
        freeze: bool = False,
        hub_entry: str = "dinov3_vits16",
    ) -> None:
        super().__init__()

        if repo_dir is None or str(repo_dir).strip() == "":
            raise ValueError("repo_dir is required for OfficialDINOv3ViTS16Backbone.")
        if weights is None or str(weights).strip() == "":
            raise ValueError("weights is required for OfficialDINOv3ViTS16Backbone.")

        from pathlib import Path

        repo_path = Path(repo_dir).expanduser().resolve()
        weight_path = Path(weights).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"DINOv3 repo_dir not found: {repo_path}")
        if not (repo_path / "hubconf.py").exists():
            raise FileNotFoundError(
                f"DINOv3 repo_dir must contain hubconf.py, got: {repo_path}. "
                "Clone official repo first: git clone https://github.com/facebookresearch/dinov3.git"
            )
        if not weight_path.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {weight_path}")

        self.repo_dir = str(repo_path)
        self.weights = str(weight_path)
        self.hub_entry = str(hub_entry)

        self.model = torch.hub.load(
            self.repo_dir,
            self.hub_entry,
            source="local",
            weights=self.weights,
        )

        # ViT-S/16 hidden dimension. Prefer model attributes when available.
        self.feature_dim = int(
            getattr(self.model, "embed_dim", None)
            or getattr(self.model, "num_features", None)
            or getattr(self.model, "hidden_size", None)
            or 384
        )

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self

    @staticmethod
    def _select_feature(out: object) -> Tensor:
        """Robustly select one global feature vector from DINOv3 outputs."""
        if isinstance(out, dict):
            # DINO-style forward_features usually exposes normalized CLS token.
            for key in (
                "x_norm_clstoken",
                "cls_token",
                "pooler_output",
                "features",
                "last_hidden_state",
                "x_prenorm",
            ):
                if key in out and torch.is_tensor(out[key]):
                    h = out[key]
                    if h.dim() == 3:   # [B, N, D]
                        h = h[:, 0, :]
                    elif h.dim() > 3:
                        h = h.flatten(1)
                    return h
            raise KeyError(f"Unknown DINOv3 output keys: {list(out.keys())}")

        if isinstance(out, (list, tuple)):
            # Prefer first tensor-like object; for [B,N,D], use CLS token.
            for item in out:
                if torch.is_tensor(item):
                    h = item
                    if h.dim() == 3:
                        h = h[:, 0, :]
                    elif h.dim() > 3:
                        h = h.flatten(1)
                    return h
                if isinstance(item, dict):
                    return OfficialDINOv3ViTS16Backbone._select_feature(item)
            raise TypeError("DINOv3 tuple/list output does not contain a tensor.")

        if torch.is_tensor(out):
            h = out
            if h.dim() == 3:
                h = h[:, 0, :]
            elif h.dim() > 3:
                h = h.flatten(1)
            return h

        # Some HF-like outputs may have attributes.
        if hasattr(out, "pooler_output") and getattr(out, "pooler_output") is not None:
            return getattr(out, "pooler_output")
        if hasattr(out, "last_hidden_state") and getattr(out, "last_hidden_state") is not None:
            h = getattr(out, "last_hidden_state")
            return h[:, 0, :] if h.dim() == 3 else h

        raise TypeError(f"Unsupported DINOv3 output type: {type(out)}")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Official DINOv3 backbones expose forward_features; using it is safer
        # because it returns CLS/patch tokens instead of task-specific heads.
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(x)
        else:
            out = self.model(x)

        h = self._select_feature(out)
        if not torch.is_tensor(h):
            raise TypeError("DINOv3 selected feature is not a tensor")
        if h.dim() != 2:
            h = h.flatten(1)
        return {"features": h}


class TimmDINOv3ViTS16Backbone(nn.Module):
    """DINOv3 ViT-S/16 feature extractor through timm.

    Model name:
        vit_small_patch16_dinov3_qkvb.lvd1689m

    This is useful when your environment prefers timm over Transformers.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3_qkvb.lvd1689m",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover
            raise ImportError("timm is required. Install with: pip install -U timm") from exc

        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = int(getattr(self.model, "num_features", 384))

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.model(x)
        if h.dim() > 2:
            h = h.flatten(1)
        return {"features": h}


BackboneName = Literal[
    "resnet50",
    "mambavision_t",
    "mambavision_s",
    "dinov3_vits16",
    "dinov3_vits16_hf",
    "dinov3_vits16_official",
    "dinov3_vits16_pth",
    "dinov3_vits16_timm",
]


def build_classification_backbone(
    name: BackboneName = "resnet50",
    *,
    pretrained: bool = False,
    cifar_stem: bool = True,
    freeze_backbone: bool = False,
    # MambaVision
    mambavision_model_path: Optional[str] = None,
    # DINOv3 HF
    dinov3_model_name_or_path: Optional[str] = None,
    dinov3_repo_dir: Optional[str] = None,
    dinov3_weights: Optional[str] = None,
    dinov3_hub_entry: str = "dinov3_vits16",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    dinov3_use_pooler_output: bool = True,
    # DINOv3 timm
    dinov3_timm_model_name: str = "vit_small_patch16_dinov3_qkvb.lvd1689m",
) -> Tuple[nn.Module, int]:
    """Build classification feature extractor.

    Returns:
        backbone:
            nn.Module returning {"features": h}
        feature_dim:
            feature dimension D
    """
    name = name.lower()

    if name == "resnet50":
        backbone = ResNet50Backbone(
            pretrained=pretrained,
            cifar_stem=cifar_stem,
        )
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.eval()
        return backbone, backbone.feature_dim

    if name == "mambavision_t":
        model_name = mambavision_model_path or "nvidia/MambaVision-T-1K"
        backbone = MambaVisionBackbone(
            model_name=model_name,
            pretrained=True,
            local_files_only=local_files_only,
            freeze=freeze_backbone,
        )
        return backbone, backbone.feature_dim

    if name == "mambavision_s":
        model_name = mambavision_model_path or "nvidia/MambaVision-S-1K"
        backbone = MambaVisionBackbone(
            model_name=model_name,
            pretrained=True,
            local_files_only=local_files_only,
            freeze=freeze_backbone,
        )
        return backbone, backbone.feature_dim


    if name in {"dinov3_vits16_official", "dinov3_vits16_pth"}:
        backbone = OfficialDINOv3ViTS16Backbone(
            repo_dir=dinov3_repo_dir or "./third_party/dinov3",
            weights=dinov3_weights or "./pretrained/dinov3_vits16_kaggle/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            freeze=freeze_backbone,
            hub_entry=dinov3_hub_entry,
        )
        return backbone, backbone.feature_dim

    if name in {"dinov3_vits16", "dinov3_vits16_hf"}:
        model_name = dinov3_model_name_or_path or "facebook/dinov3-vits16-pretrain-lvd1689m"
        backbone = DINOv3ViTS16Backbone(
            model_name_or_path=model_name,
            pretrained=True,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            freeze=freeze_backbone,
            use_pooler_output=dinov3_use_pooler_output,
        )
        return backbone, backbone.feature_dim

    if name == "dinov3_vits16_timm":
        backbone = TimmDINOv3ViTS16Backbone(
            model_name=dinov3_timm_model_name,
            pretrained=True,
            freeze=freeze_backbone,
        )
        return backbone, backbone.feature_dim

    raise ValueError(f"Unknown classification backbone: {name}")


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
