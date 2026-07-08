from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def make_balanced_indices(
    targets: Sequence[int],
    num_classes: int,
    samples_per_class: Optional[int] = None,
    seed: int = 2024,
) -> List[int]:
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    class_indices = []
    min_count = min([(targets == c).sum() for c in range(num_classes)])

    if samples_per_class is None:
        samples_per_class = int(min_count)

    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        class_indices.extend(idx[:samples_per_class].tolist())

    rng.shuffle(class_indices)
    return class_indices


def make_long_tailed_indices(
    targets: Sequence[int],
    num_classes: int,
    max_samples: Optional[int] = None,
    imbalance_factor: float = 100.0,
    seed: int = 2024,
) -> List[int]:
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    if max_samples is None:
        max_samples = min([(targets == c).sum() for c in range(num_classes)])

    # exponential long-tail: class 0 head, class C-1 tail
    counts = []
    for c in range(num_classes):
        if num_classes == 1:
            n = max_samples
        else:
            n = max_samples * (imbalance_factor ** (-c / (num_classes - 1)))
        counts.append(max(1, int(round(n))))

    selected = []
    for c, n in enumerate(counts):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        selected.extend(idx[: min(n, len(idx))].tolist())

    rng.shuffle(selected)
    return selected


def make_indices_by_counts(
    targets: Sequence[int],
    class_counts: Sequence[int],
    seed: int = 2024,
) -> List[int]:
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    selected = []
    for c, n in enumerate(class_counts):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        selected.extend(idx[: min(int(n), len(idx))].tolist())

    rng.shuffle(selected)
    return selected