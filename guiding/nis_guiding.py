"""Neural Importance Sampling (NIS) guiding distribution.

This module implements path guiding using piecewise-quadratic coupling flows,
based on the NIS paper: "Neural Importance Sampling" (Müller et al., 2019).

The NIS approach uses normalizing flows to learn the importance distribution,
with the key insight that piecewise-quadratic couplings allow:
- Exact density evaluation (for MIS weights)
- Fast sampling (via inverse CDF)
- Efficient training (via KL divergence minimization)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False

from guiding.base import GuidingDistribution, GuidingConfig
from guiding.training_data import TrainingBatch
from guiding.config import get_logger

logger = get_logger("nis_guiding")


@dataclass
class NISConfig(GuidingConfig):
    """Configuration for NIS guiding distribution."""
    # Flow architecture
    num_coupling_layers: int = 4
    num_bins: int = 32
    hidden_dim: int = 64
    
    # Conditioning network (encodes position/wi into conditioning)
    conditioning_dim: int = 64
    
    # Training
    learning_rate: float = 5e-3
    
    # Use tinycudann for acceleration
    use_tcnn: bool = True


class PiecewiseQuadraticCoupling(nn.Module):
    """Piecewise-quadratic coupling layer for normalizing flows.
    
    Implements the coupling transform from NIS paper Section 4.
    The piecewise-quadratic CDF allows exact density evaluation
    while being invertible for sampling.
    """
    
    def __init__(self, num_bins: int = 32, hidden_dim: int = 64, use_tcnn: bool = True):
        super().__init__()
        self.num_bins = num_bins
        self.n_out = num_bins + (num_bins + 1)  # Widths (K) + Vertices (K+1)
        
        if use_tcnn and TCNN_AVAILABLE:
            self._build_tcnn_network(hidden_dim)
        else:
            self._build_pytorch_network(hidden_dim)
    
    def _build_tcnn_network(self, hidden_dim: int):
        """Build network using tinycudann for speed."""
        self.use_tcnn = True
        
        encoding_config = {
            "otype": "OneBlob",
            "n_bins": self.num_bins
        }
        
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": 3
        }
        
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=1,
            n_output_dims=self.n_out,
            encoding_config=encoding_config,
            network_config=network_config
        )
    
    def _build_pytorch_network(self, hidden_dim: int):
        """Fallback PyTorch network."""
        self.use_tcnn = False
        
        # OneBlob encoding
        self.register_buffer('centers', torch.linspace(0, 1, self.num_bins))
        self.sigma = 1.0 / self.num_bins
        
        self.net = nn.Sequential(
            nn.Linear(self.num_bins, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_out)
        )
    
    def _oneblob_encode(self, x: torch.Tensor) -> torch.Tensor:
        """OneBlob positional encoding for PyTorch fallback."""
        diff = x - self.centers.unsqueeze(0)
        encoded = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        return encoded / (encoded.sum(dim=1, keepdim=True) + 1e-8)
    
    def get_params(self, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalized widths W and vertices V from conditioning input."""
        if self.use_tcnn:
            if not x_a.is_contiguous():
                x_a = x_a.contiguous()
            out = self.net(x_a).float()
        else:
            encoded = self._oneblob_encode(x_a.squeeze(-1))
            out = self.net(encoded)
        
        W_unnorm, V_unnorm = torch.split(out, [self.num_bins, self.num_bins + 1], dim=1)
        
        # Normalize widths (softmax)
        W = F.softmax(W_unnorm, dim=1)
        
        # Normalize vertices (trapezoidal area normalization)
        exp_V = torch.exp(V_unnorm.clamp(-10, 10))  # Clamp for stability
        v_left = exp_V[:, :-1]
        v_right = exp_V[:, 1:]
        trapezoid_areas = 0.5 * (v_left + v_right) * W
        total_area = torch.sum(trapezoid_areas, dim=1, keepdim=True)
        V = exp_V / (total_area + 1e-8)
        
        return W, V
    
    def forward(self, x_b: torch.Tensor, W: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transform: x -> z (data to latent), returns (z, log_det)."""
        widths_cum = torch.cumsum(W, dim=1)
        pad = torch.zeros((W.shape[0], 1), device=W.device)
        edges = torch.cat([pad, widths_cum], dim=1)
        edges[:, -1] = 1.0
        
        val = x_b.clamp(0.0, 0.9999)
        
        # Find bin index
        bin_idx = torch.sum(val >= edges, dim=1, keepdim=True) - 1
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)
        
        # Gather parameters for the bin
        w_b = torch.gather(W, 1, bin_idx)
        v_b = torch.gather(V, 1, bin_idx)
        v_b1 = torch.gather(V, 1, bin_idx + 1)
        edge_b = torch.gather(edges, 1, bin_idx)
        
        # Relative position in bin
        alpha = (val - edge_b) / (w_b + 1e-8)
        
        # Cumulative area before this bin
        v_all_left = V[:, :-1]
        v_all_right = V[:, 1:]
        areas = 0.5 * (v_all_left + v_all_right) * W
        cum_areas = torch.cumsum(areas, dim=1)
        cum_areas_pad = torch.cat([pad, cum_areas], dim=1)
        area_pre = torch.gather(cum_areas_pad, 1, bin_idx)
        
        # Quadratic CDF within bin
        term1 = alpha * v_b * w_b
        term2 = 0.5 * (alpha ** 2) * (v_b1 - v_b) * w_b
        z = area_pre + term1 + term2
        
        # Log determinant (log PDF)
        pdf_val = v_b + alpha * (v_b1 - v_b)
        log_det = torch.log(pdf_val + 1e-8)
        
        return z, log_det
    
    def inverse(self, z_b: torch.Tensor, W: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Inverse transform: z -> x (latent to data), for sampling."""
        widths_cum = torch.cumsum(W, dim=1)
        pad = torch.zeros((W.shape[0], 1), device=W.device)
        edges = torch.cat([pad, widths_cum], dim=1)
        edges[:, -1] = 1.0
        
        target_area = z_b.clamp(0.0, 0.9999)
        
        # Compute cumulative areas
        v_all_left = V[:, :-1]
        v_all_right = V[:, 1:]
        bin_areas = 0.5 * (v_all_left + v_all_right) * W
        cum_areas = torch.cumsum(bin_areas, dim=1)
        cum_areas_edges = torch.cat([pad, cum_areas], dim=1)
        cum_areas_edges[:, -1] = 1.0
        
        # Find bin
        bin_idx = torch.sum(target_area >= cum_areas_edges, dim=1, keepdim=True) - 1
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)
        
        # Gather parameters
        w_b = torch.gather(W, 1, bin_idx)
        v_b = torch.gather(V, 1, bin_idx)
        v_b1 = torch.gather(V, 1, bin_idx + 1)
        edge_area = torch.gather(cum_areas_edges, 1, bin_idx)
        edge_x = torch.gather(edges, 1, bin_idx)
        
        # Solve quadratic: A*alpha^2 + B*alpha + C = 0
        A = 0.5 * (v_b1 - v_b) * w_b
        B = v_b * w_b
        C = edge_area - target_area
        
        det = (B ** 2 - 4 * A * C).clamp(min=0.0)
        sqrt_det = torch.sqrt(det)
        
        alpha_quad = (-B + sqrt_det) / (2 * A + 1e-10)
        alpha_linear = -C / (B + 1e-10)
        
        is_quadratic = torch.abs(A) > 1e-6
        alpha = torch.where(is_quadratic, alpha_quad, alpha_linear).clamp(0, 1)
        
        return edge_x + alpha * w_b


class ConditioningNetwork(nn.Module):
    """Network that encodes (position, wi, roughness) into conditioning for the flow."""
    
    def __init__(self, output_dim: int = 64, use_tcnn: bool = True):
        super().__init__()
        # Input: position (3) + wi (3) + roughness (1) = 7
        input_dim = 7
        
        if use_tcnn and TCNN_AVAILABLE:
            config = {
                "encoding": {
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,  # position
                            "otype": "HashGrid",
                            "n_levels": 8,
                            "n_features_per_level": 2,
                            "log2_hashmap_size": 17,
                            "base_resolution": 16,
                            "per_level_scale": 1.5,
                        },
                        {
                            "n_dims_to_encode": 4,  # wi + roughness
                            "otype": "Identity",
                        }
                    ]
                },
                "network": {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            }
            self.net = tcnn.NetworkWithInputEncoding(
                n_input_dims=input_dim,
                n_output_dims=output_dim,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
            self.use_tcnn = True
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )
            self.use_tcnn = False
    
    def forward(self, position: torch.Tensor, wi: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        """Encode scene context into conditioning vector."""
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)
        
        x = torch.cat([position, wi, roughness], dim=1)
        
        if self.use_tcnn:
            x = x.contiguous()
        
        return self.net(x).float()


class NISFlow(nn.Module):
    """Neural Importance Sampling flow for 2D directional sampling.
    
    Uses piecewise-quadratic coupling layers to learn a mapping from
    uniform [0,1]^2 to the target importance distribution on the hemisphere.
    
    The 2D parameterization uses spherical coordinates (theta, phi) mapped to [0,1]^2.
    """
    
    def __init__(self, config: NISConfig):
        super().__init__()
        self.config = config
        
        # Conditioning network
        self.conditioning = ConditioningNetwork(
            output_dim=config.conditioning_dim,
            use_tcnn=config.use_tcnn
        )
        
        # Coupling layers - alternate which dimension is transformed
        self.layers = nn.ModuleList()
        for i in range(config.num_coupling_layers):
            # Each layer gets conditioning input + the "fixed" dimension
            self.layers.append(
                PiecewiseQuadraticCoupling(
                    num_bins=config.num_bins,
                    hidden_dim=config.hidden_dim,
                    use_tcnn=config.use_tcnn
                )
            )
            # Register mask as buffer
            mask = torch.tensor([1.0, 0.0]) if i % 2 == 0 else torch.tensor([0.0, 1.0])
            self.register_buffer(f'mask_{i}', mask)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute log probability of x given condition.
        
        Args:
            x: Points in [0,1]^2 (N, 2)
            condition: Conditioning from ConditioningNetwork (N, conditioning_dim)
            
        Returns:
            log_prob: Log probability (N,)
        """
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for i, layer in enumerate(self.layers):
            mask = getattr(self, f'mask_{i}')
            
            idx_A = torch.nonzero(mask).squeeze()
            idx_B = torch.nonzero(1 - mask).squeeze()
            
            x_a = z[:, idx_A].unsqueeze(1)  # Conditioning dimension
            x_b = z[:, idx_B].unsqueeze(1)  # Transformed dimension
            
            # Get flow parameters from conditioning dim
            W, V = layer.get_params(x_a)
            
            # Transform
            y_b, log_det = layer.forward(x_b, W, V)
            
            # Update
            z_new = z.clone()
            z_new[:, idx_B] = y_b.squeeze(1)
            z = z_new
            log_det_sum = log_det_sum + log_det.squeeze()
        
        return log_det_sum
    
    def sample(self, num_samples: int, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the flow given conditioning.
        
        Args:
            num_samples: Number of samples (should match condition batch size)
            condition: Conditioning from ConditioningNetwork (N, conditioning_dim)
            
        Returns:
            samples: Points in [0,1]^2 (N, 2)
            log_prob: Log probability of samples (N,)
        """
        device = condition.device
        z = torch.rand((num_samples, 2), device=device)
        
        # Inverse pass through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            mask = getattr(self, f'mask_{i}')
            
            idx_A = torch.nonzero(mask).squeeze()
            idx_B = torch.nonzero(1 - mask).squeeze()
            
            x_a = z[:, idx_A].unsqueeze(1)
            z_b = z[:, idx_B].unsqueeze(1)
            
            W, V = layer.get_params(x_a)
            x_b = layer.inverse(z_b, W, V)
            
            z_new = z.clone()
            z_new[:, idx_B] = x_b.squeeze(1)
            z = z_new
        
        # Compute log prob of samples
        log_prob = self.forward(z, condition)
        
        return z, log_prob


class NISGuidingDistribution(GuidingDistribution):
    """Path guiding using Neural Importance Sampling.
    
    Uses normalizing flows to learn the importance distribution directly,
    rather than approximating with a mixture model (like vMF).
    
    Advantages over vMF:
    - More expressive (can represent arbitrary distributions)
    - Better for complex lighting (caustics, intricate shadows)
    
    Disadvantages:
    - Slower training and inference
    - More parameters
    """
    
    def __init__(self, config: Optional[NISConfig] = None):
        config = config or NISConfig()
        super().__init__(config)
        self.nis_config = config
        
        # Build the flow
        self.flow = NISFlow(config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.flow.parameters(),
            lr=config.learning_rate
        )
        
        logger.info(f"NIS Guiding Distribution initialized with {config.num_coupling_layers} layers")
    
    @property
    def name(self) -> str:
        return "Neural Importance Sampling"
    
    def _direction_to_uv(self, direction: torch.Tensor) -> torch.Tensor:
        """Convert 3D direction to [0,1]^2 UV coordinates.
        
        Uses equal-area spherical mapping for uniform sampling.
        """
        # Normalize direction
        direction = F.normalize(direction, dim=-1)
        
        # Spherical coordinates
        # theta = arccos(z), phi = atan2(y, x)
        theta = torch.acos(direction[:, 2].clamp(-1, 1))  # [0, pi]
        phi = torch.atan2(direction[:, 1], direction[:, 0])  # [-pi, pi]
        
        # Map to [0, 1]
        u = theta / torch.pi
        v = (phi + torch.pi) / (2 * torch.pi)
        
        return torch.stack([u, v], dim=1)
    
    def _uv_to_direction(self, uv: torch.Tensor) -> torch.Tensor:
        """Convert [0,1]^2 UV coordinates to 3D direction."""
        u, v = uv[:, 0], uv[:, 1]
        
        theta = u * torch.pi
        phi = v * 2 * torch.pi - torch.pi
        
        sin_theta = torch.sin(theta)
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = torch.cos(theta)
        
        return torch.stack([x, y, z], dim=1)
    
    def _uv_jacobian(self, uv: torch.Tensor) -> torch.Tensor:
        """Jacobian of the UV -> direction mapping (for PDF correction)."""
        u = uv[:, 0]
        theta = u * torch.pi
        # Jacobian is sin(theta) * pi * 2*pi = 2*pi^2 * sin(theta)
        return 2 * (torch.pi ** 2) * torch.sin(theta).clamp(min=1e-8)
    
    def sample(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample directions from the NIS flow."""
        self.flow.eval()
        
        with torch.no_grad():
            # Get conditioning
            condition = self.flow.conditioning(position, wi, roughness)
            
            # Sample UV coordinates
            uv, log_prob_uv = self.flow.sample(position.shape[0], condition)
            
            # Convert to directions
            directions = self._uv_to_direction(uv)
            
            # Convert log prob (account for Jacobian)
            jacobian = self._uv_jacobian(uv)
            pdf_values = torch.exp(log_prob_uv) / jacobian
            
        return directions, pdf_values
    
    # Alias for compatibility with PathGuidingIntegrator
    def sample_guided_direction(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for sample() for API compatibility."""
        return self.sample(position, wi, roughness)
    
    def pdf(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate PDF for given directions."""
        self.flow.eval()
        
        with torch.no_grad():
            # Get conditioning
            condition = self.flow.conditioning(position, wi, roughness)
            
            # Convert direction to UV
            uv = self._direction_to_uv(wo)
            
            # Get log prob
            log_prob_uv = self.flow.forward(uv, condition)
            
            # Convert to PDF (account for Jacobian)
            jacobian = self._uv_jacobian(uv)
            pdf = torch.exp(log_prob_uv) / jacobian
            
        return pdf
    
    def log_pdf(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate log PDF for given directions."""
        self.flow.eval()
        
        with torch.no_grad():
            condition = self.flow.conditioning(position, wi, roughness)
            uv = self._direction_to_uv(wo)
            log_prob_uv = self.flow.forward(uv, condition)
            jacobian = self._uv_jacobian(uv)
            return log_prob_uv - torch.log(jacobian)
    
    def train_step(self, batch: TrainingBatch) -> float:
        """Train the NIS flow using importance-weighted KL divergence."""
        if not batch.is_valid():
            return -1.0
        
        # Prepare data
        pos = batch.position_normalized * 2 - 1  # [0,1] -> [-1,1]
        wo = batch.wo
        wi = batch.wi
        roughness = batch.roughness
        radiance = batch.radiance_rgb.mean(dim=1)  # Luminance as importance weight
        
        self.flow.train()
        
        # Get conditioning
        condition = self.flow.conditioning(pos, wi, roughness)
        
        # Convert directions to UV
        uv = self._direction_to_uv(wo)
        
        # Forward pass to get log probability
        log_prob = self.flow.forward(uv, condition)
        
        # Account for Jacobian
        jacobian = self._uv_jacobian(uv)
        log_pdf = log_prob - torch.log(jacobian)
        
        # Importance-weighted negative log-likelihood
        # Weight by radiance (higher radiance = more important to model correctly)
        weight = radiance.clamp(max=10.0)  # Clamp to prevent outliers
        loss = -(weight * log_pdf).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    # Alias for compatibility with visualize.py
    def train_step_from_batch(self, batch: TrainingBatch) -> float:
        """Alias for train_step for API compatibility."""
        return self.train_step(batch)
    
    def get_distribution_for_visualization(
        self,
        position: torch.Tensor,
        wi: torch.Tensor,
        roughness: torch.Tensor
    ) -> "NISGuidingDistribution":
        """Return self - visualization uses pdf() directly."""
        # Store context for pdf calls during visualization
        self._vis_position = position
        self._vis_wi = wi
        self._vis_roughness = roughness
        return self


# Remove NISVisualization class - not needed anymore
