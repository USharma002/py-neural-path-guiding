"""Neural Radiance Cache for accelerating path tracing."""
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import drjit as dr
import tinycudann as tcnn
import torch
import torch.nn as nn

from guiding.config import get_logger
from guiding.training_data import TrainingBatch

logger = get_logger("nrc")


class NeuralRadianceCache(nn.Module):
    """Neural network for caching and predicting radiance values.
    
    Implements the architecture from 'Neural Path Guiding with Distribution Factorization' (2025).
    
    Inputs:
    - Position (HashGrid)
    - View Direction (Spherical Harmonics)
    - Normal (Spherical Harmonics)
    - Roughness (OneBlob)
    
    Target:
    - RGB Radiance (Relative L2 Loss)
    """
    def __init__(self, device: str = "cuda", learning_rate: float = 1e-2) -> None:
        super().__init__()
        self.device = device

        # Inputs: Pos(3) + View(3) + Normal(3) + Roughness(1) = 10 dims
        input_dims = 10
        output_dims = 3  # RGB
        
        config = {
            "encoding": {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3, # Position
                        "otype": "HashGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": 16,
                        "per_level_scale": 1.5,
                    },
                    {
                        "n_dims_to_encode": 3, # View Direction (batch.wi)
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    {
                        "n_dims_to_encode": 3, # Normal (batch.normal)
                        "otype": "SphericalHarmonics",
                        "degree": 4, 
                    },
                    {
                        "n_dims_to_encode": 1, # Roughness (batch.roughness)
                        "otype": "OneBlob", # Specified by paper
                        "n_bins": 4,
                    }
                ]
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 4,
            }
        }

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=input_dims,
            n_output_dims=output_dims,
            encoding_config=config["encoding"],
            network_config=config["network"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-15)
        
        logger.info("Neural Radiance Cache initialized (Paper Config).")

    def _normalize_position(self, pos: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
        """Normalize position to [0, 1]^3 for hash grid encoding."""
        # Add epsilon to avoid division by zero
        return (pos - bbox_min) / (bbox_max - bbox_min + 1e-6)

    def _prepare_input(
        self, 
        positions: torch.Tensor, 
        view_dirs: torch.Tensor,
        normals: torch.Tensor, 
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate inputs matching the composite encoding order."""
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)

        # Normalize directions to be safe (SH requires unit vectors)
        # TCNN expects inputs to be contiguous float
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        # Order: Pos(3), View(3), Normal(3), Rough(1)
        return torch.cat([positions, view_dirs, normals, roughness], dim=1).float().contiguous()

    def query(
        self, 
        positions: torch.Tensor, 
        normals: torch.Tensor, 
        view_dirs: torch.Tensor, 
        roughness: torch.Tensor,
        bbox_min: Optional[torch.Tensor] = None,
        bbox_max: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inference query (used by Integrator to get cached radiance)."""
        self.model.eval()
        
        # Normalize positions if bounds are provided
        if bbox_min is not None and bbox_max is not None:
            positions = self._normalize_position(positions, bbox_min, bbox_max)
        else:
            # Fallback/Safety: Clamp to [0, 1] if already normalized
            positions = torch.clamp(positions, 0.0, 1.0)
        
        with torch.no_grad():
            input_tensor = self._prepare_input(positions, view_dirs, normals, roughness)
            raw_output = self.model(input_tensor.to(self.device))
        
        # Softplus for positive energy
        return torch.nn.functional.softplus(raw_output.float())

    def forward(
        self, 
        positions: torch.Tensor, 
        view_dirs: torch.Tensor,
        normals: torch.Tensor, 
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Training forward pass."""
        input_tensor = self._prepare_input(positions, view_dirs, normals, roughness)
        raw_output = self.model(input_tensor)
        return torch.nn.functional.softplus(raw_output.float())

    def train_step_from_batch(self, batch: TrainingBatch) -> float:
        """Train using data from the shared TrainingBatch."""
        if not batch.is_valid():
            return -1.0
        
        # Data mapping:
        # batch.position_normalized -> Position Input
        # batch.wi                  -> View Direction Input (Incoming from camera)
        # batch.normal              -> Normal Input
        # batch.roughness           -> Roughness Input
        # batch.radiance_rgb        -> Target
        
        self.model.train()
        
        # 1. Forward Pass
        pred = self.forward(
            batch.position_normalized, 
            batch.wi, 
            batch.normal, 
            batch.roughness
        )
        
        # 2. Relative L2 Loss (Unbiased for MC integration)
        # L = (pred - target)^2 / (pred^2 + epsilon)
        # Note: We detach the denominator to treat it as a constant weight per sample.
        # This prevents the gradient from encouraging the network to simply increase 
        # predictions to minimize the denominator.
        targets = batch.radiance_rgb
        denom = pred.detach().square() + 1e-2
        loss = torch.mean(((pred - targets).square()) / denom)

        # 3. Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()