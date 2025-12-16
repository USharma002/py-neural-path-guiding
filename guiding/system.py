"""Path guiding system that trains and queries the neural guiding network."""
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, Any

import drjit as dr
import torch
import logging

from guiding.config import get_logger, TrainingConfig
from utils.math_utils import M_EPSILON
from guiding.training_data import TrainingBatch, prepare_shared_training_data
import guiding.registry as registry

if TYPE_CHECKING:
    from integrators.path_guiding_integrator import PathGuidingIntegrator

logger = get_logger("guiding")


class PathGuidingSystem:
    """Main path guiding system that coordinates training and inference.
    
    Position normalization is delegated to individual distributions.
    Each distribution receives world-space positions and normalizes as needed.
    """
    
    def __init__(
        self, 
        device: str = "cuda", 
        config: Optional[TrainingConfig] = None,
        method_name: Optional[str] = None,
        **kwargs
    ) -> None:
        self.device = device
        self.config = config or TrainingConfig()
        self.scene  = None

        self._method_name = method_name or registry.get_default_method()
        self._distribution = self._create_distribution(device, **kwargs)
        
        # Vis buffers
        self._vis_position = None
        self._vis_wi = None
        self._vis_roughness = None
        
        logger.info(f"Path Guiding System initialized with {self._distribution.name}")

    def set_scene(self, scene: Any) -> None:
        """Pass the Mitsuba scene to the system and underlying distribution."""
        self.scene = scene
        self._distribution.set_scene(scene)

    def _create_distribution(self, device: str, **kwargs):
        return registry.create_distribution(self._method_name, device, **kwargs)
    
    def switch_method(self, method_name: str, **kwargs) -> None:
        if method_name == self._method_name:
            logger.info(f"Already using {method_name}")
            return
        
        logger.info(f"Switching guiding method from {self._method_name} to {method_name}")
        self._method_name = method_name
        self._distribution = registry.create_distribution(method_name, self.device, **kwargs)
        if self.scene is not None:
            self._distribution.set_scene(self.scene)
    
    @property
    def method_name(self) -> str:
        return self._method_name

    def _log_data_stats(self, label: str, tensor: torch.Tensor) -> None:
        """Helper to log min/max/mean of a tensor for debugging."""
        if logger.isEnabledFor(logging.INFO):
            with torch.no_grad():
                t_min = tensor.min(dim=0)[0]
                t_max = tensor.max(dim=0)[0]
                l_min = [f"{x:.3f}" for x in t_min.cpu().tolist()]
                l_max = [f"{x:.3f}" for x in t_max.cpu().tolist()]
                logger.info(f"[{label}] Min: {l_min}, Max: {l_max}")

    def train_step(self, integrator: "PathGuidingIntegrator") -> float:
        """Perform one training step using data from the integrator.
        
        The batch contains world-space positions. The distribution
        handles normalization internally.
        """
        # 1. Use the shared function to get a validated batch (world space positions)
        batch = prepare_shared_training_data(integrator, device=self.device)

        # 2. Check validity
        if not batch.is_valid():
            logger.warning("Skipping training step due to no valid data (width=0 or all filtered).")
            return -1.0
        
        # 4. Train - distribution handles normalization
        return self._distribution.train_step(batch)

    def train_step_from_batch(self, batch: TrainingBatch) -> float:
        """Train from a pre-prepared TrainingBatch."""
        if not batch.is_valid():
            return -1.0
        return self._distribution.train_step(batch)

    def pdf(
        self, 
        position: torch.Tensor, 
        wi: torch.Tensor, 
        wo: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Compute the PDF of the guiding distribution.
        
        Args:
            position: (N, 3) positions in WORLD SPACE
        """
        # Distribution handles normalization internally
        return self._distribution.pdf(position, wi, wo, roughness)
    
    def get_distribution_for_visualization(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> "PathGuidingSystem":
        """Return self for visualization.
        
        Args:
            position: (1, 3) position in WORLD SPACE
        """
        # Store world space - distribution will normalize when needed
        self._vis_position = position
        self._vis_wi = wi
        self._vis_roughness = roughness
        return self

    def sample_guided_direction(
        self, 
        position: torch.Tensor, 
        wi: torch.Tensor, 
        roughness: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample directions from the trained guiding distribution.
        
        Args:
            position: (N, 3) positions in WORLD SPACE
        """
        # Distribution handles normalization internally
        return self._distribution.sample(position, wi, roughness)
