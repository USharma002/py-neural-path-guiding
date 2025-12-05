"""Path guiding system that trains and queries the neural guiding network.

This module provides the main PathGuidingSystem class which coordinates:
- Data preparation from the integrator
- Training of the underlying distribution
- Sampling and PDF evaluation for rendering

The actual distribution (VMF, NIS, etc.) is pluggable via composition.
Use guiding_registry to register new guiding methods.
"""
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, Union

import drjit as dr
import torch

from guiding.config import get_logger, TrainingConfig
from utils.math_utils import M_EPSILON
from guiding.training_data import TrainingBatch
import guiding.registry as registry

if TYPE_CHECKING:
    from integrators.path_guiding_integrator import PathGuidingIntegrator

logger = get_logger("guiding")


class PathGuidingSystem:
    """Main path guiding system that coordinates training and inference.
    
    This class handles:
    - Data preparation and validation from the integrator
    - Bounding box normalization
    - Delegation to the underlying distribution for sample/pdf/train
    
    The actual distribution is created via guiding_registry,
    keeping a clean interface for the integrator.
    
    Args:
        device: Device to run computations on
        config: Training configuration
        method_name: Name of the guiding method (from registry)
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
        
        # Store the method name for potential switching
        self._method_name = method_name or registry.get_default_method()
        
        # Bounding box for position normalization (set by integrator)
        self.bbox_min: Optional[torch.Tensor] = None
        self.bbox_max: Optional[torch.Tensor] = None
        
        # Create the underlying distribution from registry
        self._distribution = self._create_distribution(device, **kwargs)
        
        # For visualization compatibility
        self._vis_position: Optional[torch.Tensor] = None
        self._vis_wi: Optional[torch.Tensor] = None
        self._vis_roughness: Optional[torch.Tensor] = None
        
        logger.info(f"Path Guiding System initialized with {self._distribution.name}")

    def _create_distribution(self, device: str, **kwargs):
        """Create the underlying guiding distribution from registry."""
        return registry.create_distribution(self._method_name, device, **kwargs)
    
    def switch_method(self, method_name: str, **kwargs) -> None:
        """Switch to a different guiding method.
        
        Args:
            method_name: Name of the new method (must be registered)
            **kwargs: Additional arguments for the new distribution
        """
        if method_name == self._method_name:
            logger.info(f"Already using {method_name}")
            return
        
        logger.info(f"Switching guiding method from {self._method_name} to {method_name}")
        self._method_name = method_name
        self._distribution = registry.create_distribution(method_name, self.device, **kwargs)
    
    @property
    def method_name(self) -> str:
        """Get the current guiding method name."""
        return self._method_name

    def _normalize_position(self, pos: torch.Tensor) -> torch.Tensor:
        """Normalize world position to [0, 1]^3 range for distribution."""
        if self.bbox_min is None or self.bbox_max is None:
            return pos
        return (pos - self.bbox_min) / (self.bbox_max - self.bbox_min + 1e-8)

    def prepare_training_data(
        self, 
        integrator: "PathGuidingIntegrator"
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Process integrator's record to produce clean, validated training tensors.
        
        Args:
            integrator: The path guiding integrator with surface interaction data
            
        Returns:
            Tuple of (pos, wo, wi, roughness, targets_li, combined_pdf) or 
            tuple of Nones if no valid data
        """
        rec = integrator.surfaceInteractionRecord
        integrator.scatter_data_into_buffer()

        if dr.width(rec.position) == 0:
            return (None,) * 6

        pos = rec.position.torch()
        wi = rec.wi.torch()
        wo = rec.wo.torch()

        pos = self._normalize_position(pos)

        targets_li = rec.radiance.torch()
        combined_pdf = rec.woPdf.torch()

        # Filter invalid samples
        valid_mask = combined_pdf > M_EPSILON
        valid_mask &= ~torch.any(torch.isinf(pos) | torch.isnan(pos), dim=1)
        valid_mask &= ~torch.any(torch.isinf(wo) | torch.isnan(wo), dim=1)
        valid_mask &= ~torch.any(torch.isinf(wi) | torch.isnan(wi), dim=1)

        wi_len_sq = torch.sum(wi * wi, dim=1)
        valid_mask &= (wi_len_sq > 0.9) & (wi_len_sq < 1.1)

        if not torch.any(valid_mask):
            return (None,) * 6

        pos = pos[valid_mask]
        wi = wi[valid_mask]
        wo = wo[valid_mask]
        targets_li = targets_li[valid_mask]
        combined_pdf = combined_pdf[valid_mask]

        roughness = torch.ones((pos.shape[0], 1), device=self.device)

        return pos, wo, wi, roughness, targets_li, combined_pdf

    def train_step(self, integrator: "PathGuidingIntegrator") -> float:
        """Perform one training step using data from the integrator.
        
        Args:
            integrator: The path guiding integrator with training data
            
        Returns:
            Loss value, or -1.0 if training was skipped
        """
        pos, wo, wi, roughness, targets_li, combined_pdf = self.prepare_training_data(integrator)

        if pos is None:
            logger.warning("Skipping training step due to no valid data.")
            return -1.0

        # Create batch and delegate to distribution
        batch = TrainingBatch(
            position_normalized=pos,
            wo=wo,
            wi=wi,
            roughness=roughness,
            radiance_rgb=targets_li,
            combined_pdf=combined_pdf
        )
        return self._distribution.train_step(batch)

    def train_step_from_batch(self, batch: TrainingBatch) -> float:
        """Train from a pre-prepared TrainingBatch.
        
        Args:
            batch: TrainingBatch from prepare_shared_training_data()
            
        Returns:
            Loss value, or -1.0 if training was skipped
        """
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
        """Compute the PDF of the guiding distribution for given directions."""
        position = self._normalize_position(position)
        return self._distribution.pdf(position, wi, wo, roughness)
    
    def get_distribution_for_visualization(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> "PathGuidingSystem":
        """Return self for visualization - pdf() will be called directly."""
        self._vis_position = self._normalize_position(position)
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
            position: World positions
            wi: Incoming directions  
            roughness: Surface roughness values
            
        Returns:
            Tuple of (sampled_directions, pdf_values)
        """
        position = self._normalize_position(position)
        return self._distribution.sample(position, wi, roughness)
