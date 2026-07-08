from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class AIDLTJsonDataset(Dataset):
    def __init__(self, split_json, split="train", transform=None):
        self.split_json = Path(split_json)
        self.transform = transform

        with self.split_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        self.meta = meta
        self.root = Path(meta["aid_root"])

        if split == "train":
            self.samples = meta["train_samples"]
        elif split in {"test", "val"}:
            self.samples = meta["test_samples"]
        else:
            raise ValueError(f"split must be train/test/val, got {split}")

        self.num_classes = meta["num_classes"]
        self.class_names = meta["class_names"]
        self.class_counts = meta["class_counts"]
        self.class_prior = meta["class_prior"]
        self.head_classes = meta["head_classes"]
        self.medium_classes = meta["medium_classes"]
        self.tail_classes = meta["tail_classes"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        path = self.root / item["path"]
        label = int(item["label"])

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label