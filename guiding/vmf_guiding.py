"""VMF (von Mises-Fisher) mixture guiding distribution.

This module implements path guiding using a mixture of vMF distributions,
following the approach from "Neural Path Guiding" papers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Any, Optional

import torch
import torch.nn as nn

from guiding.base import GuidingDistribution, GuidingConfig
from networks.guiding_network import GuidingNetwork
from distributions.vmf_mixture import BatchedMixedSphericalGaussianDistribution
from guiding.training_data import TrainingBatch
from guiding.config import get_logger

logger = get_logger("vmf_guiding")


@dataclass
class VMFConfig(GuidingConfig):
    """Configuration for VMF mixture guiding."""
    # Number of vMF lobes in the mixture
    num_lobes: int = 8
    
    # Training hyperparameters
    learning_rate: float = 5e-3
    importance_clipping_threshold: float = 10.0
    importance_weight_max: float = 100.0
    kappa_regularization_strength: float = 0.001
    epsilon: float = 1e-7
    
    # Gradient clipping
    gradient_clip_value: float = 1.0
    gradient_clip_norm: float = 1.0


class VMFGuidingDistribution(GuidingDistribution):
    """Path guiding using mixture of von Mises-Fisher distributions.
    
    The network predicts parameters for K vMF lobes:
    - lambda (K): mixture weights
    - kappa (K): concentration parameters  
    - mu (K, 3): mean directions (from theta, phi)
    
    Training uses importance-weighted MLE loss.
    """
    
    def __init__(self, config: Optional[VMFConfig] = None):
        config = config or VMFConfig()
        super().__init__(config)
        self.vmf_config = config
        
        # Neural network for predicting vMF parameters
        self.network = GuidingNetwork(
            device=self.device,
            K=config.num_lobes
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.network.model.parameters(),
            lr=config.learning_rate
        )
        
        logger.info(f"VMF Guiding Distribution initialized with {config.num_lobes} lobes")
    
    @property
    def name(self) -> str:
        return "VMF Mixture"
    
    def sample(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample directions from the vMF mixture.
        
        Args:
            position: Normalized positions in [0,1]^3
            wi: Incoming directions
            roughness: Surface roughness
        """
        self.network.eval()
        
        # Convert [0,1] -> [-1,1] for network
        pos = position * 2 - 1
        
        with torch.no_grad():
            params = self.network(pos, wi, roughness)
            dist = BatchedMixedSphericalGaussianDistribution(params)
            directions = dist.sample()
            pdf_values = dist.pdf(directions)
            
        return directions, pdf_values
    
    def pdf(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate PDF for given directions.
        
        Args:
            position: Normalized positions in [0,1]^3
        """
        self.network.eval()
        
        # Convert [0,1] -> [-1,1] for network
        pos = position * 2 - 1
        
        with torch.no_grad():
            params = self.network(pos, wi, roughness)
            dist = BatchedMixedSphericalGaussianDistribution(params)
            return dist.pdf(wo)
    
    def log_pdf(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate log PDF for given directions."""
        pdf_val = self.pdf(position, wi, wo, roughness)
        return torch.log(pdf_val + self.vmf_config.epsilon)
    
    def train_step(self, batch: TrainingBatch) -> float:
        """Train the vMF network using importance-weighted MLE."""
        if not batch.is_valid():
            return -1.0
        
        # Normalize position to [-1, 1]
        pos = batch.position_normalized * 2 - 1  # [0,1] -> [-1,1]
        wo = batch.wo
        wi = batch.wi
        roughness = batch.roughness
        targets_li = batch.radiance_rgb
        combined_pdf = batch.combined_pdf
        
        self.network.train()
        
        # Forward pass - get predicted distribution
        params = self.network(pos, wi, roughness)
        kappas = params['kappa']
        dist = BatchedMixedSphericalGaussianDistribution(params)
        
        # Compute importance-weighted MLE loss
        ground_truth_radiance = targets_li.mean(dim=1)
        ground_truth_radiance = ground_truth_radiance.clamp(max=self.vmf_config.importance_clipping_threshold)
        
        # PDF of sampled direction under our distribution
        prob = dist.pdf(wo)
        epsilon = self.vmf_config.epsilon
        
        # Importance weight
        importance_weight = ground_truth_radiance.detach() / (combined_pdf + epsilon)
        importance_weight = importance_weight.clamp(max=self.vmf_config.importance_weight_max)
        
        # MLE loss with importance weighting
        log_prob = torch.log(prob + epsilon)
        mle_loss = -importance_weight * log_prob
        
        # Kappa regularization to prevent concentration collapse
        kappa_loss = self.vmf_config.kappa_regularization_strength * torch.mean(kappas.pow(2))
        
        loss = torch.mean(mle_loss) + kappa_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_value_(
            self.network.model.parameters(),
            clip_value=self.vmf_config.gradient_clip_value
        )
        torch.nn.utils.clip_grad_norm_(
            self.network.model.parameters(),
            max_norm=self.vmf_config.gradient_clip_norm
        )
        
        self.optimizer.step()
        
        return loss.item()
    
    def get_distribution_for_visualization(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> "VMFGuidingDistribution":
        """Return self - visualization uses pdf() directly."""
        # Store context for pdf calls during visualization
        self._vis_position = position
        self._vis_wi = wi
        self._vis_roughness = roughness
        return self
