"""Abstract base class for guiding distributions used in path guiding."""
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
    Each distribution normalizes positions according to its own needs using the scene bbox.
    """
    
    def __init__(self, config: Optional[GuidingConfig] = None):
        self.config = config or GuidingConfig()
        self.device = self.config.device
        
        # Scene reference for bbox-based normalization
        self.scene: Optional[Any] = None
        self.bbox_min: Optional[torch.Tensor] = None
        self.bbox_max: Optional[torch.Tensor] = None

    def set_scene(self, scene: Any) -> None:
        """
        Set the Mitsuba scene and extract bounding box for normalization.
        
        Implementations requiring ray tracing (like VXPG) should override this
        and call super().set_scene(scene) to get bbox setup.
        """
        self.scene = scene
        if scene is not None:
            bbox = scene.bbox()
            
            # Convert Mitsuba ScalarPoint3f to list then to torch tensor
            # ScalarPoint3f doesn't support DLPack, so we extract components manually
            bbox_min_list = [bbox.min[i] for i in range(3)]
            bbox_max_list = [bbox.max[i] for i in range(3)]
            
            self.bbox_min = torch.tensor(bbox_min_list, dtype=torch.float32, device=self.device)
            self.bbox_max = torch.tensor(bbox_max_list, dtype=torch.float32, device=self.device)
    
    @abstractmethod
    def sample(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample directions from the guiding distribution.
        
        Args:
            position: Surface positions (N, 3), in WORLD SPACE
            wi: Incoming directions (N, 3), normalized
            roughness: Surface roughness (N, 1)
            
        Returns:
            Tuple of:
            - directions: (N, 3) sampled unit vectors in local coordinates
            - pdf_values: (N,) probability densities
            
        NOTE: Implementation should call self._normalize_position(position) as needed
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
            position: Surface positions (N, 3), in WORLD SPACE
            wi: Incoming directions (N, 3)
            wo: Outgoing directions to evaluate (N, 3)
            roughness: Surface roughness (N, 1)
            
        Returns:
            (N,) probability density values
            
        NOTE: Implementation should call self._normalize_position(position) as needed
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
            position: Surface positions (N, 3), in WORLD SPACE
            wi: Incoming directions (N, 3)
            wo: Outgoing directions to evaluate (N, 3)
            roughness: Surface roughness (N, 1)
            
        Returns:
            (N,) log probability density values
            
        NOTE: Implementation should call self._normalize_position(position) as needed
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: TrainingBatch) -> float:
        """Perform one training step.
        
        Args:
            batch: TrainingBatch with positions in WORLD SPACE
            
        Returns:
            Loss value for this step
            
        NOTE: Implementation should call self._normalize_position(batch.position) as needed
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
            position: Single position (1, 3), in WORLD SPACE
            wi: Single incoming direction (1, 3)
            roughness: Single roughness value (1, 1)
            
        Returns:
            Distribution-specific visualization object
            
        NOTE: Implementation should call self._normalize_position(position) as needed
        """
        pass

    def map_to_voxel_index(self, position_01: torch.Tensor, resolution: int) -> torch.Tensor:
        """Helper to map [0,1] positions to voxel indices [0, res-1]."""
        # Clamp to ensure 1.0 doesn't overflow to resolution
        pos_clamped = torch.clamp(position_01, 0.0, 0.999999)
        return (pos_clamped * resolution).long()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the distribution."""
        pass
