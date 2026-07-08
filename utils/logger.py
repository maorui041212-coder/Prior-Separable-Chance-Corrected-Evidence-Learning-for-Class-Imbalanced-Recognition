"""
Logging utilities for CCEL-Net.

Provides:
    - setup_logger: text log to console and file
    - JsonlLogger: structured jsonl metric logger
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "ccel",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger that writes to both terminal and file.

    Args:
        name:
            Logger name.
        log_file:
            Path to text log file. If None, only prints to terminal.
        level:
            Logging level.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicated handlers when called multiple times.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class JsonlLogger:
    """
    JSONL logger for epoch-level metrics.

    Each call writes one JSON object per line.
    """

    def __init__(self, path: str | Path, mode: str = "a") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, mode, encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        if self.f.closed:
            raise RuntimeError(
                f"JsonlLogger file is already closed: {self.path}. "
                "Check whether json_logger.close() was called inside the training loop."
            )

        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        if not self.f.closed:
            self.f.close()