"""Abstract base class for guiding distributions used in path guiding.

This module defines the interface that any guiding distribution must implement.
Concrete implementations (VMF, NIS, etc.) handle their own:
- Network architecture
- Parameter handling  
- Training logic
- Sampling logic

To add a new distribution:
1. Subclass GuidingDistribution
2. Implement all abstract methods
3. Import and use in PathGuidingSystem
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn

from guiding.training_data import TrainingBatch


@dataclass
class GuidingConfig:
    """Base configuration for guiding distributions."""
    device: str = "cuda"
    learning_rate: float = 1e-3
    

class GuidingDistribution(ABC):
    """Abstract base class for all guiding distributions.
    
    Any distribution used for path guiding must implement:
    - sample(): Draw directional samples given position/incoming direction
    - pdf(): Evaluate probability density for given directions
    - log_pdf(): Evaluate log probability density (for stable training)
    - train_step(): Perform one training step given a batch of data
    
    The distribution is conditioned on scene context (position, incoming direction, etc.)
    """
    
    def __init__(self, config: Optional[GuidingConfig] = None):
        self.config = config or GuidingConfig()
        self.device = self.config.device
        
        # Bounding box for position normalization (set by integrator)
        self.bbox_min: Optional[torch.Tensor] = None
        self.bbox_max: Optional[torch.Tensor] = None
    
    @abstractmethod
    def sample(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample directions from the guiding distribution.
        
        Args:
            position: Surface positions (N, 3), normalized to [-1,1]^3
            wi: Incoming directions (N, 3), normalized
            roughness: Surface roughness (N, 1)
            
        Returns:
            Tuple of:
            - directions: (N, 3) sampled unit vectors in local coordinates
            - pdf_values: (N,) probability densities
        """
        pass
    
    @abstractmethod
    def pdf(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the probability density for given directions.
        
        Args:
            position: Surface positions (N, 3)
            wi: Incoming directions (N, 3)
            wo: Outgoing directions to evaluate (N, 3)
            roughness: Surface roughness (N, 1)
            
        Returns:
            (N,) probability density values
        """
        pass
    
    @abstractmethod
    def log_pdf(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the log probability density for given directions.
        
        Args:
            position: Surface positions (N, 3)
            wi: Incoming directions (N, 3)
            wo: Outgoing directions to evaluate (N, 3)
            roughness: Surface roughness (N, 1)
            
        Returns:
            (N,) log probability density values
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: TrainingBatch) -> float:
        """Perform one training step.
        
        Args:
            batch: TrainingBatch with all necessary training data
            
        Returns:
            Loss value for this step
        """
        pass
    
    @abstractmethod
    def get_distribution_for_visualization(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> Any:
        """Get a distribution object for visualization purposes.
        
        This returns whatever representation is useful for visualizing
        the distribution (e.g., vMF mixture parameters, flow samples, etc.)
        
        Args:
            position: Single position (1, 3)
            wi: Single incoming direction (1, 3)
            roughness: Single roughness value (1, 1)
            
        Returns:
            Distribution-specific visualization object
        """
        pass
    
    def _normalize_position(self, pos: torch.Tensor) -> torch.Tensor:
        """Normalize world position to [-1, 1]^3 range."""
        if self.bbox_min is None or self.bbox_max is None:
            return pos
        return 2 * ((pos - self.bbox_min) / (self.bbox_max - self.bbox_min + 1e-8)) - 1
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the distribution."""
        pass
