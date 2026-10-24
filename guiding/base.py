"""Abstract base class for guiding distributions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from guiding.training_data import TrainingBatch
from utils.math_utils import normalize_position, scene_bbox_tensors


@dataclass
class GuidingConfig:
    device: str = "cuda"
    # Overridden by subclasses; declared here so it is listed first in the settings dialog.
    learning_rate: float = 1e-3


class GuidingDistribution(ABC):
    """Directional distribution conditioned on (position, wi, normal, roughness).

    Positions and normals are in world space; wi and the sampled/evaluated directions are in the
    local shading frame.
    """

    def __init__(self, config: Optional[GuidingConfig] = None):
        self.config = config or GuidingConfig()
        self.device = self.config.device
        self.bbox_min: Optional[torch.Tensor] = None
        self.bbox_max: Optional[torch.Tensor] = None

    def set_scene(self, scene: Any) -> None:
        self.bbox_min, self.bbox_max = scene_bbox_tensors(scene, self.device)

    def _position_to_unit_cube(self, position: torch.Tensor) -> torch.Tensor:
        return normalize_position(position, self.bbox_min, self.bbox_max)

    @abstractmethod
    def sample(self, position: torch.Tensor, wi: torch.Tensor, normal: torch.Tensor,
               roughness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (directions (N, 3), pdf (N,))."""

    @abstractmethod
    def pdf(self, position: torch.Tensor, wi: torch.Tensor, normal: torch.Tensor,
            roughness: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        """Return the solid-angle density of wo, shape (N,)."""

    @abstractmethod
    def train_step(self, batch: TrainingBatch) -> float:
        """Run one optimization step and return the loss."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
