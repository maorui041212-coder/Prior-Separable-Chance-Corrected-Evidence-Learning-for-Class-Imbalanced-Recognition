"""
Cityscapes-MICE dataset builders for pixel-level imbalanced segmentation.

Recommended placement:
    ccel/datasets/cityscapes_mice.py

Supported dataset roots, for example:
    /data2/mr/MICE/CASWiT/dataset/Cityscapes-MICE-traffic_sign-bg995_fg005-patch128_allsplits
    /data2/mr/MICE/CASWiT/dataset/Cityscapes-MICE-3class_trafficsign_trafficlight-bg70_fg30-patch128_allsplits

Expected directory layouts. The loader will automatically try these forms:

    root/train/images       root/train/masks
    root/val/images         root/val/masks
    root/test/images        root/test/masks

or:

    root/images/train       root/masks/train
    root/images/val         root/masks/val
    root/images/test        root/masks/test

It also accepts common aliases such as image/imgs and mask/masks/labels/gt.
Masks should be single-channel label maps with class ids 0..C-1 and optional
ignore label 255.

The dataset object provides:
    - num_classes
    - class_names
    - class_counts          pixel counts, excluding ignore_index
    - get_class_counts_tensor()
    - get_class_priors_tensor()
    - get_tail_class_ids()

These statistics can be shared by CE, Weighted CE, Logit Adjustment, MICELoss,
CCEL-Net, and segmentation metrics.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


try:
    from ccel.utils.seed import seed_worker, build_generator
except Exception:  # pragma: no cover - fallback for standalone use
    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def build_generator(seed: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(int(seed))
        return g


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CLASS_NAMES: Dict[int, List[str]] = {
    2: ["background", "traffic_sign"],
    3: ["background", "traffic_sign", "traffic_light"],
    4: ["background", "traffic_sign", "traffic_light", "person"],
    5: ["background", "traffic_sign", "traffic_light", "person", "car"],
    10: [
        "background",
        "traffic_sign",
        "traffic_light",
        "person",
        "rider",
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
    ],
}


# -----------------------------------------------------------------------------
# Path discovery
# -----------------------------------------------------------------------------


def infer_num_classes_from_name(root: Union[str, Path], default: Optional[int] = None) -> int:
    """Infer num_classes from Cityscapes-MICE folder name."""
    name = Path(root).name.lower()
    if "10class" in name:
        return 10
    if "5class" in name:
        return 5
    if "4class" in name:
        return 4
    if "3class" in name:
        return 3
    if "traffic_sign" in name or "trafficsign" in name:
        return 2
    if default is not None:
        return int(default)
    raise ValueError(
        f"Cannot infer num_classes from root name: {Path(root).name}. "
        "Please pass num_classes explicitly."
    )


def default_class_names(num_classes: int) -> List[str]:
    if num_classes in DEFAULT_CLASS_NAMES:
        return list(DEFAULT_CLASS_NAMES[num_classes])
    return [f"class_{i}" for i in range(num_classes)]


def _canonical_key(path: Path) -> str:
    """Make image/mask names comparable across common Cityscapes suffixes."""
    stem = path.stem
    suffixes = [
        "_leftImg8bit",
        "_gtFine_labelIds",
        "_gtFine_labelTrainIds",
        "_gtFine_color",
        "_mask",
        "_masks",
        "_label",
        "_labels",
        "_gt",
        "_seg",
    ]
    changed = True
    while changed:
        changed = False
        for s in suffixes:
            if stem.endswith(s):
                stem = stem[: -len(s)]
                changed = True
    return stem


def _collect_files(directory: Path, exts: Iterable[str]) -> List[Path]:
    exts = {e.lower() for e in exts}
    if not directory.exists():
        return []
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _candidate_split_dirs(root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_names = ["images", "image", "imgs", "img", "leftImg8bit", "rgb"]
    mask_names = ["masks", "mask", "labels", "label", "gt", "gts", "annotations", "ann", "targets", "target"]

    pairs: List[Tuple[Path, Path]] = []

    # root/train/images + root/train/masks
    for img_name in image_names:
        for mask_name in mask_names:
            pairs.append((root / split / img_name, root / split / mask_name))

    # root/images/train + root/masks/train
    for img_name in image_names:
        for mask_name in mask_names:
            pairs.append((root / img_name / split, root / mask_name / split))

    # root/train/img + root/train/label etc.; also case split names use validation
    alt_split = {"val": ["valid", "validation"], "test": ["testing"]}.get(split, [])
    for s in alt_split:
        for img_name in image_names:
            for mask_name in mask_names:
                pairs.append((root / s / img_name, root / s / mask_name))
                pairs.append((root / img_name / s, root / mask_name / s))

    return pairs


def find_image_mask_pairs(root: Union[str, Path], split: str) -> List[Tuple[Path, Path]]:
    """Find paired image/mask files for one split."""
    root = Path(root)
    split = split.lower()

    # 1) Text split file support: train.txt lines can be either "img mask" or
    # one stem per line. Relative paths are resolved from root.
    for txt_name in [f"{split}.txt", f"{split}_list.txt", f"{split}_pairs.txt"]:
        txt = root / txt_name
        if txt.exists():
            pairs: List[Tuple[Path, Path]] = []
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        img = Path(parts[0])
                        mask = Path(parts[1])
                        if not img.is_absolute():
                            img = root / img
                        if not mask.is_absolute():
                            mask = root / mask
                        pairs.append((img, mask))
                    else:
                        # stem-only fallback is handled by directory scanning below.
                        pass
            if pairs:
                missing = [(i, m) for i, m in pairs if not i.exists() or not m.exists()]
                if missing:
                    raise FileNotFoundError(f"Some pairs in {txt} do not exist. Example: {missing[0]}")
                return pairs

    # 2) Directory scanning.
    for img_dir, mask_dir in _candidate_split_dirs(root, split):
        img_files = _collect_files(img_dir, IMG_EXTS)
        mask_files = _collect_files(mask_dir, MASK_EXTS)
        if not img_files or not mask_files:
            continue

        mask_by_key: Dict[str, Path] = {}
        for m in mask_files:
            key = _canonical_key(m)
            mask_by_key.setdefault(key, m)

        pairs = []
        for img in img_files:
            key = _canonical_key(img)
            mask = mask_by_key.get(key)
            if mask is not None:
                pairs.append((img, mask))

        if pairs:
            return pairs

    tried = "\n".join([f"  images={a} masks={b}" for a, b in _candidate_split_dirs(root, split)[:12]])
    raise FileNotFoundError(
        f"Cannot find image/mask pairs for split={split!r} under root={root}.\n"
        f"Tried common layouts such as:\n{tried}\n"
        "Please make sure the folder names are images/masks or pass a supported layout."
    )


# -----------------------------------------------------------------------------
# Paired transforms
# -----------------------------------------------------------------------------


class CityscapesMICETransform:
    """Paired image/mask transform for segmentation."""

    def __init__(
        self,
        train: bool,
        image_size: Optional[Union[int, Tuple[int, int]]] = 128,
        hflip_prob: float = 0.5,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.train = bool(train)
        self.image_size = image_size
        self.hflip_prob = float(hflip_prob)
        self.mean = tuple(float(x) for x in mean)
        self.std = tuple(float(x) for x in std)

    @staticmethod
    def _resolve_size(image_size: Optional[Union[int, Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        if image_size is None:
            return None
        if isinstance(image_size, int):
            return (int(image_size), int(image_size))
        if len(image_size) != 2:
            raise ValueError("image_size must be int, tuple(H, W), or None")
        return (int(image_size[0]), int(image_size[1]))

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")

        size = self._resolve_size(self.image_size)
        if size is not None:
            image = TF.resize(image, size=size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, size=size, interpolation=TF.InterpolationMode.NEAREST)

        if self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)

        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            # Most generated Cityscapes-MICE masks should be single-channel or palette.
            # If an RGB image is accidentally provided, use the first channel rather
            # than converting colors to class ids silently.
            mask_np = mask_np[..., 0]
        mask_tensor = torch.from_numpy(mask_np.astype(np.int64, copy=False)).long()
        return image_tensor, mask_tensor


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


@dataclass
class CityscapesMICEMeta:
    dataset_name: str
    split: str
    num_classes: int
    class_names: List[str]
    num_samples: int
    class_counts: List[int]
    class_priors: List[float]
    ignore_index: Optional[int]


class CityscapesMICEDataset(Dataset):
    """Cityscapes-MICE patch-level segmentation dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = 128,
        ignore_index: Optional[int] = 255,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        compute_class_counts: bool = True,
        cache_class_counts: bool = True,
        return_meta: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split.lower()
        self.num_classes = int(num_classes) if num_classes is not None else infer_num_classes_from_name(self.root)
        self.class_names = list(class_names) if class_names is not None else default_class_names(self.num_classes)
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")

        self.image_size = image_size
        self.ignore_index = ignore_index
        self.return_meta = bool(return_meta)
        self.pairs = find_image_mask_pairs(self.root, self.split)

        self.transform = transform
        if self.transform is None:
            self.transform = CityscapesMICETransform(
                train=(self.split == "train"),
                image_size=image_size,
            )

        if compute_class_counts:
            self.class_counts = self._load_or_compute_class_counts(cache=cache_class_counts)
        else:
            self.class_counts = [0 for _ in range(self.num_classes)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image_tensor, mask_tensor = self.transform(image, mask)

        if self.return_meta:
            meta = {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "index": int(index),
            }
            return image_tensor, mask_tensor, meta
        return image_tensor, mask_tensor

    @property
    def dataset_name(self) -> str:
        return self.root.name

    def _cache_path(self) -> Path:
        return self.root / f".cityscapes_mice_{self.split}_C{self.num_classes}_counts.json"

    def _load_or_compute_class_counts(self, cache: bool = True) -> List[int]:
        cache_path = self._cache_path()
        if cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                counts = data.get("class_counts", None)
                if isinstance(counts, list) and len(counts) == self.num_classes:
                    return [int(x) for x in counts]
            except Exception:
                pass

        counts = self.compute_pixel_class_counts()
        if cache:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "dataset_name": self.dataset_name,
                            "split": self.split,
                            "num_classes": self.num_classes,
                            "class_names": self.class_names,
                            "class_counts": counts,
                            "ignore_index": self.ignore_index,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception:
                pass
        return counts

    def compute_pixel_class_counts(self) -> List[int]:
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for _, mask_path in self.pairs:
            mask = Image.open(mask_path)
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            y = torch.from_numpy(mask_np.astype(np.int64, copy=False)).reshape(-1).long()
            if self.ignore_index is not None:
                y = y[y != int(self.ignore_index)]
            valid = (y >= 0) & (y < self.num_classes)
            y = y[valid]
            if y.numel() > 0:
                counts += torch.bincount(y, minlength=self.num_classes).cpu()
        return [int(x) for x in counts.tolist()]

    def get_class_counts_tensor(self) -> torch.Tensor:
        return torch.tensor(self.class_counts, dtype=torch.float32)

    def get_class_priors_tensor(self) -> torch.Tensor:
        counts = self.get_class_counts_tensor()
        return counts / counts.sum().clamp_min(1.0)

    def get_tail_class_ids(
        self,
        *,
        exclude_background: bool = True,
        background_index: int = 0,
        top_k: Optional[int] = None,
        prior_threshold: Optional[float] = None,
    ) -> List[int]:
        """
        Return rare/tail class ids according to train pixel priors.

        Default:
            all non-background classes sorted from rare to frequent.

        With top_k:
            only return the rarest top_k classes.

        With prior_threshold:
            only return classes whose prior <= threshold.
        """
        priors = self.get_class_priors_tensor()
        candidates = list(range(self.num_classes))
        if exclude_background and 0 <= background_index < self.num_classes:
            candidates = [c for c in candidates if c != background_index]
        if prior_threshold is not None:
            candidates = [c for c in candidates if float(priors[c]) <= float(prior_threshold)]
        candidates = sorted(candidates, key=lambda c: float(priors[c]))
        if top_k is not None:
            candidates = candidates[: int(top_k)]
        return candidates

    def meta(self) -> CityscapesMICEMeta:
        priors = self.get_class_priors_tensor().tolist()
        return CityscapesMICEMeta(
            dataset_name=self.dataset_name,
            split=self.split,
            num_classes=self.num_classes,
            class_names=self.class_names,
            num_samples=len(self),
            class_counts=[int(x) for x in self.class_counts],
            class_priors=[float(x) for x in priors],
            ignore_index=self.ignore_index,
        )

    def save_meta(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = self.meta().__dict__
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------


def build_cityscapes_mice_datasets(
    root: Union[str, Path],
    *,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    image_size: Optional[Union[int, Tuple[int, int]]] = 128,
    ignore_index: Optional[int] = 255,
    compute_class_counts: bool = True,
    cache_class_counts: bool = True,
    return_meta: bool = False,
) -> Tuple[CityscapesMICEDataset, CityscapesMICEDataset, CityscapesMICEDataset]:
    train_set = CityscapesMICEDataset(
        root=root,
        split="train",
        num_classes=num_classes,
        class_names=class_names,
        image_size=image_size,
        ignore_index=ignore_index,
        compute_class_counts=compute_class_counts,
        cache_class_counts=cache_class_counts,
        return_meta=return_meta,
    )
    resolved_num_classes = train_set.num_classes
    resolved_class_names = train_set.class_names

    val_set = CityscapesMICEDataset(
        root=root,
        split="val",
        num_classes=resolved_num_classes,
        class_names=resolved_class_names,
        image_size=image_size,
        ignore_index=ignore_index,
        compute_class_counts=compute_class_counts,
        cache_class_counts=cache_class_counts,
        return_meta=return_meta,
    )
    test_set = CityscapesMICEDataset(
        root=root,
        split="test",
        num_classes=resolved_num_classes,
        class_names=resolved_class_names,
        image_size=image_size,
        ignore_index=ignore_index,
        compute_class_counts=compute_class_counts,
        cache_class_counts=cache_class_counts,
        return_meta=return_meta,
    )
    return train_set, val_set, test_set


def build_cityscapes_mice_loaders(
    root: Union[str, Path],
    *,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    image_size: Optional[Union[int, Tuple[int, int]]] = 128,
    ignore_index: Optional[int] = 255,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 2024,
    pin_memory: bool = True,
    drop_last_train: bool = True,
    compute_class_counts: bool = True,
    cache_class_counts: bool = True,
    return_meta: bool = False,
):
    train_set, val_set, test_set = build_cityscapes_mice_datasets(
        root=root,
        num_classes=num_classes,
        class_names=class_names,
        image_size=image_size,
        ignore_index=ignore_index,
        compute_class_counts=compute_class_counts,
        cache_class_counts=cache_class_counts,
        return_meta=return_meta,
    )

    generator = build_generator(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader, test_loader, train_set, val_set, test_set


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def print_dataset_summary(dataset: CityscapesMICEDataset) -> None:
    priors = dataset.get_class_priors_tensor()
    print("=" * 80)
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Split: {dataset.split}")
    print(f"Num samples: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")
    print(f"Ignore index: {dataset.ignore_index}")
    print("Class pixel counts / priors:")
    for c, name in enumerate(dataset.class_names):
        print(f"  {c:02d} {name:>16s}: count={dataset.class_counts[c]} prior={float(priors[c]):.8f}")
    print(f"Tail class ids excluding background: {dataset.get_tail_class_ids(exclude_background=True)}")


def save_all_split_meta(
    train_set: CityscapesMICEDataset,
    val_set: CityscapesMICEDataset,
    test_set: CityscapesMICEDataset,
    out_dir: Union[str, Path],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_set.save_meta(out_dir / "train_meta.json")
    val_set.save_meta(out_dir / "val_meta.json")
    test_set.save_meta(out_dir / "test_meta.json")


__all__ = [
    "CityscapesMICETransform",
    "CityscapesMICEDataset",
    "CityscapesMICEMeta",
    "build_cityscapes_mice_datasets",
    "build_cityscapes_mice_loaders",
    "print_dataset_summary",
    "save_all_split_meta",
    "infer_num_classes_from_name",
    "default_class_names",
]
