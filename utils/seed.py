"""
Random seed utilities for CCEL-Net.

This module centralizes random seed control for:
    - Python random
    - NumPy
    - PyTorch CPU / CUDA
    - cuDNN behavior
    - DataLoader workers
    - DataLoader generator

Recommended usage in training scripts:

    from ccel.utils.seed import set_seed, seed_worker, build_generator

    set_seed(seed=2024, deterministic=False)

    generator = build_generator(seed)

    DataLoader(
        dataset,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(
    seed: int = 2024,
    deterministic: bool = False,
    benchmark: Optional[bool] = None,
) -> None:
    """
    Set random seed for reproducible experiments.

    Args:
        seed:
            Global random seed.

        deterministic:
            If True, enable deterministic algorithms when possible.
            This improves reproducibility but may slow down training.

        benchmark:
            Controls torch.backends.cudnn.benchmark.

            If None:
                benchmark = not deterministic

            For most training experiments:
                deterministic=False is recommended.

            For strict reproducibility checks:
                deterministic=True is recommended.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if benchmark is None:
        benchmark = not deterministic

    torch.backends.cudnn.benchmark = bool(benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """
    Seed function for PyTorch DataLoader workers.

    Usage:
        DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed: int = 2024) -> torch.Generator:
    """
    Build a torch.Generator for DataLoader shuffle control.

    Usage:
        generator = build_generator(seed)

        DataLoader(
            dataset,
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker,
        )
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def get_rng_state() -> dict:
    """
    Save current RNG states into checkpoint.

    This is useful if you want strict resume training behavior.
    """
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda_random_state_all"] = None

    return state


def set_rng_state(state: dict) -> None:
    """
    Restore RNG states from checkpoint.

    Use this only when resuming training and the checkpoint contains RNG state.
    """
    if not state:
        return

    if "python_random_state" in state:
        random.setstate(state["python_random_state"])

    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])

    if "torch_random_state" in state:
        torch.set_rng_state(state["torch_random_state"])

    cuda_state = state.get("torch_cuda_random_state_all", None)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)