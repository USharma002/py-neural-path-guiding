"""Logging, viewer settings and an FPS counter."""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

import torch

torch.backends.cuda.matmul.allow_tf32 = True

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s", datefmt="%H:%M:%S"))
logging.getLogger("npg").addHandler(_handler)
logging.getLogger("npg").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"npg.{name}")


@dataclass
class VisualizerConfig:
    main_resolution: tuple[int, int] = (256, 256)
    hover_resolution: tuple[int, int] = (256, 256)
    realtime_interval_ms: int = 100
    fps_update_interval: float = 0.5  # seconds


class FPSProfiler:
    """Duration of the last frame."""

    def __init__(self):
        self._start = 0.0
        self._elapsed = 0.0

    def begin_frame(self) -> None:
        self._start = time.perf_counter()

    def end_frame(self) -> None:
        self._elapsed = time.perf_counter() - self._start

    def get_stats_string(self) -> str:
        if self._elapsed <= 0:
            return "FPS: --"
        return f"FPS: {1 / self._elapsed:.1f} ({self._elapsed * 1000:.1f}ms)"
