"""Distribution Factorization (DF) guiding distribution.

Based on: "Neural Path Guiding with Distribution Factorization" (Figueiredo et al., 2025).
Reference: https://github.com/pedrovfigueiredo/guiding-df

Key Features:
- Factorization: p(u, v) = p(u) * p(v|u)
- Mapping: Area-preserving square-to-sphere mapping (Uniform Jacobian).
- Encodings: SH for direction, OneBlob for roughness, Triangle Wave for conditional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import math

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

logger = get_logger("df_guiding")


@dataclass
class DFConfig(GuidingConfig):
    """Configuration matching the paper's parameters."""
    # Method
    interpolation_mode: Literal["nearest", "linear"] = "linear" # Paper DF-L
    
    # Grid Resolution (Paper: M1=32, M2=16)
    resolution_u: int = 32  
    resolution_v: int = 16 
    
    # Architecture
    hidden_dim: int = 64
    n_hidden_layers: int = 3
    
    # Conditional Encoding (Paper: Triangle Wave with 12 frequencies)
    u_encoding_freqs: int = 12 
    
    # Training
    learning_rate: float = 1e-2 # Paper: 1e-2
    use_tcnn: bool = True


class TriangleWaveEncoding(nn.Module):
    """Triangle Wave Encoding as used in NRC and this paper for epsilon_1."""
    def __init__(self, n_frequencies=12):
        super().__init__()
        self.n_frequencies = n_frequencies
        # Frequencies: 2^0, 2^1, ... 2^(k-1)
        self.register_buffer("freqs", 2.0 ** torch.arange(n_frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0, 1]
        # T_k(x) = | 2(2^k x mod 1) - 1 |
        # In PyTorch: triangle_wave(v) = 2 * abs(v - floor(v + 0.5)) ? 
        # Or simply: abs(2 * (x * freq % 1) - 1)
        
        # x: [B, 1] -> [B, n_freqs]
        scaled = x * self.freqs
        # Triangle wave in range [-1, 1]
        val = 2.0 * torch.abs(2.0 * (scaled - torch.floor(scaled + 0.5))) - 1.0
        return val


class ContextEncoder(nn.Module):
    """
    Encodes scene context (Position, Direction, Roughness).
    Paper specs:
    - Position: Learnable Dense Grid (We use HashGrid as efficient equivalent)
    - Direction: Spherical Harmonics (Degree 4)
    - Roughness: OneBlob (4 bins)
    """
    def __init__(self, output_dim: int = 64, use_tcnn: bool = True):
        super().__init__()
        
        # Input: Pos(3) + Dir(3) + Rough(1)
        if use_tcnn and TCNN_AVAILABLE:
            config = {
                "encoding": {
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3, # Position
                            "otype": "HashGrid",
                            "n_levels": 12,
                            "n_features_per_level": 2,
                            "log2_hashmap_size": 17,
                            "base_resolution": 16
                        },
                        {
                            "n_dims_to_encode": 3, # Direction (Omega_o)
                            "otype": "SphericalHarmonics",
                            "degree": 4 
                        },
                        {
                            "n_dims_to_encode": 1, # Roughness
                            "otype": "OneBlob",
                            "n_bins": 4 
                        }
                    ]
                },
                "network": {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 3
                }
            }
            self.net = tcnn.NetworkWithInputEncoding(
                n_input_dims=7,
                n_output_dims=output_dim,
                encoding_config=config["encoding"],
                network_config=config["network"]
            )
        else:
            # Fallback (Simpler MLP)
            self.net = nn.Sequential(
                nn.Linear(7, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, output_dim)
            )
            
    def forward(self, pos, dir, rough):
        if rough.ndim == 1: rough = rough.unsqueeze(1)
        # Ensure inputs are in valid ranges for encodings
        # Pos is assumed normalized to [0,1] or bound
        # Dir should be normalized
        # Rough in [0,1]
        x = torch.cat([pos, dir, rough], dim=1)
        return self.net(x).float()


class DFGuidingDistribution(GuidingDistribution):
    def __init__(self, config: Optional[DFConfig] = None):
        config = config or DFConfig()
        super().__init__(config)
        self.df_config = config
        
        # 1. Context Encoder
        self.context_net = ContextEncoder(
            output_dim=config.hidden_dim, # Context feature size
            use_tcnn=config.use_tcnn
        ).to(self.device)
        
        # 2. U-Encoding (Triangle Wave for conditional input)
        self.u_encoder = TriangleWaveEncoding(n_frequencies=config.u_encoding_freqs).to(self.device)
        u_feat_dim = config.u_encoding_freqs # Triangle wave output size
        
        # 3. PDF Estimators (Heads)
        # Marginal Head: Context(64) -> M1
        self.marginal_head = nn.Linear(config.hidden_dim, config.resolution_u).to(self.device)
        
        # Conditional Head: Context(64) + U_Feats(12) -> MLP -> M2
        # The paper uses a small network for the conditional part taking context+u
        cond_input_dim = config.hidden_dim + u_feat_dim
        
        if config.use_tcnn and TCNN_AVAILABLE:
            self.conditional_net = tcnn.Network(
                n_input_dims=cond_input_dim,
                n_output_dims=config.resolution_v,
                network_config={
                    "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                    "n_neurons": config.hidden_dim, "n_hidden_layers": 2 # Slightly smaller head
                }
            ).to(self.device)
        else:
            self.conditional_net = nn.Sequential(
                nn.Linear(cond_input_dim, config.hidden_dim), nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(),
                nn.Linear(config.hidden_dim, config.resolution_v)
            ).to(self.device)

        # Optimization
        params = (
            list(self.context_net.parameters()) + 
            list(self.marginal_head.parameters()) + 
            list(self.conditional_net.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        
        logger.info(f"DF Guiding ({config.interpolation_mode.upper()}) initialized. "
                    f"Res: {config.resolution_u}x{config.resolution_v}")

    @property
    def name(self) -> str:
        return f"DF-{self.df_config.interpolation_mode[0].upper()}"

    # =========================================================================
    #  Mapping Logic (Paper: Area Preserving)
    # =========================================================================
    
    def _direction_to_uv(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Paper Mapping:
        phi = 2 * pi * epsilon_1
        theta = arccos(1 - 2 * epsilon_2)  (Mapping z from 1..-1 to 0..1)
        """
        dir_norm = F.normalize(direction, dim=-1)
        x, y, z = dir_norm[:, 0], dir_norm[:, 1], dir_norm[:, 2]
        
        # Phi [-pi, pi] -> [0, 1]
        phi = torch.atan2(y, x)
        u = (phi + torch.pi) / (2 * torch.pi) # epsilon_1
        
        # Theta: z = cos(theta). 
        # We want v (epsilon_2) such that z = 1 - 2v  =>  2v = 1 - z  => v = (1 - z) / 2
        # z in [-1, 1]. if z=1 (pole), v=0. if z=-1, v=1.
        v = (1.0 - z) / 2.0
        
        return torch.stack([u, v], dim=1).clamp(0.0, 1.0)

    def _uv_to_direction(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Inverse of above.
        """
        u, v = uv[:, 0], uv[:, 1]
        
        # Phi
        phi = u * (2 * torch.pi) - torch.pi
        
        # Z
        z = 1.0 - 2.0 * v
        
        # Sin theta (for x, y)
        # sin^2 + cos^2 = 1 => sin = sqrt(1 - z^2)
        sin_theta = torch.sqrt((1.0 - z**2).clamp(min=0.0))
        
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        
        return torch.stack([x, y, z], dim=1)

    def _jacobian(self) -> float:
        """
        For the area preserving mapping:
        d_omega = 4 * pi * du * dv
        p(omega) = p(u, v) / (4 * pi)
        Jacobian is constant 4 * pi.
        """
        return 4 * torch.pi

    # =========================================================================
    #  Sampling / Eval Utils
    # =========================================================================

    def _get_density(self, logits: torch.Tensor, M: int) -> torch.Tensor:
        # Softmax to probs, multiply by M to get density
        return M * F.softmax(logits, dim=1)

    def _eval_1d(self, density: torch.Tensor, coords: torch.Tensor, M: int) -> torch.Tensor:
        # coords in [0, 1]
        if self.df_config.interpolation_mode == "nearest":
            idx = (coords * M).long().clamp(0, M - 1)
            return torch.gather(density, 1, idx)
        else:
            # Linear (DF-L)
            # Center of bin i is at (i + 0.5) / M
            m = coords * M - 0.5
            idx_floor = torch.floor(m).long().clamp(0, M - 1)
            idx_ceil = (idx_floor + 1).clamp(0, M - 1)
            alpha = (m - torch.floor(m)).clamp(0, 1)
            
            val_l = torch.gather(density, 1, idx_floor)
            val_r = torch.gather(density, 1, idx_ceil)
            return (1 - alpha) * val_l + alpha * val_r

    def _sample_1d(self, density: torch.Tensor, rand: torch.Tensor, M: int) -> torch.Tensor:
        B = density.shape[0]
        
        # Construct CDF
        if self.df_config.interpolation_mode == "nearest":
            # Piecewise constant -> Linear CDF
            areas = density * (1.0 / M)
            cdf = torch.cumsum(areas, dim=1)
            cdf = cdf / (cdf[:, -1:] + 1e-8)
            cdf_full = torch.cat([torch.zeros(B, 1, device=self.device), cdf], dim=1)
            
            idx = torch.searchsorted(cdf_full, rand) - 1
            idx = idx.clamp(0, M - 1)
            
            cdf_lo = torch.gather(cdf_full, 1, idx)
            cdf_hi = torch.gather(cdf_full, 1, idx+1)
            local_u = (rand - cdf_lo) / (cdf_hi - cdf_lo + 1e-8)
            return (idx + local_u.clamp(0,1)) / M
            
        else:
            # Piecewise linear -> Quadratic CDF
            # Trapezoids between bin centers (Simplified DF-L sampling)
            v_curr = density
            v_next = torch.cat([density[:, 1:], density[:, -1:]], dim=1)
            
            bin_width = 1.0 / M
            areas = 0.5 * (v_curr + v_next) * bin_width
            cdf = torch.cumsum(areas, dim=1)
            cdf = cdf / (cdf[:, -1:] + 1e-8)
            cdf_full = torch.cat([torch.zeros(B, 1, device=self.device), cdf], dim=1)
            
            idx = torch.searchsorted(cdf_full, rand) - 1
            idx = idx.clamp(0, M - 1)
            
            # Quadratic solve
            cdf_lo = torch.gather(cdf_full, 1, idx)
            y0 = torch.gather(v_curr, 1, idx)
            y1 = torch.gather(v_next, 1, idx)
            delta = y1 - y0
            
            target = rand - cdf_lo
            
            # Area(t) = width * (y0*t + 0.5*delta*t^2)
            # We solve for normalized t in [0,1]
            # Since we are in normalized CDF space, we scale by bin_area
            bin_area = torch.gather(cdf_full, 1, idx+1) - cdf_lo
            frac = target / (bin_area + 1e-10) # 0..1
            
            # (y0*t + 0.5*delta*t^2) / (y0 + 0.5*delta) = frac
            A = 0.5 * delta
            B = y0
            C = -frac * (y0 + 0.5 * delta)
            
            det = (B**2 - 4*A*C).clamp(min=0)
            sqrt_det = torch.sqrt(det)
            
            t_quad = (-B + sqrt_det) / (2*A + 1e-10)
            t_lin = -C / (B + 1e-10)
            
            is_lin = torch.abs(A) < 1e-5
            t = torch.where(is_lin, t_lin, t_quad).clamp(0, 1)
            
            return (idx + t) / M

    # =========================================================================
    #  Main Interface
    # =========================================================================

    def sample(self, position, wi, roughness):
        self.context_net.eval()
        self.conditional_net.eval()
        
        B = position.shape[0]
        with torch.no_grad():
            # 1. Context
            context = self.context_net(position, wi, roughness) # [B, 64]
            
            # 2. Sample U (Marginal)
            logits_u = self.marginal_head(context)
            pdf_u_vals = self._get_density(logits_u, self.df_config.resolution_u)
            
            rand_u = torch.rand(B, 1, device=self.device)
            u_samples = self._sample_1d(pdf_u_vals, rand_u, self.df_config.resolution_u)
            
            # 3. Sample V (Conditional)
            # Encode U for conditional net
            u_feats = self.u_encoder(u_samples) # [B, 12]
            cond_input = torch.cat([context, u_feats], dim=1)
            
            logits_v = self.conditional_net(cond_input).float()
            pdf_v_vals = self._get_density(logits_v, self.df_config.resolution_v)
            
            rand_v = torch.rand(B, 1, device=self.device)
            v_samples = self._sample_1d(pdf_v_vals, rand_v, self.df_config.resolution_v)
            
            # 4. Combine
            uv = torch.cat([u_samples, v_samples], dim=1)
            directions = self._uv_to_direction(uv)
            
            # 5. PDF Calculation
            val_u = self._eval_1d(pdf_u_vals, u_samples, self.df_config.resolution_u)
            val_v = self._eval_1d(pdf_v_vals, v_samples, self.df_config.resolution_v)
            
            pdf_uv = val_u * val_v
            
            # Jacobian correction
            # p(omega) = p(uv) / 4pi
            pdf_dir = pdf_uv / self._jacobian()
            
            return directions, pdf_dir.squeeze()

    def pdf(self, position, wi, wo, roughness):
        self.context_net.eval()
        self.conditional_net.eval()
        
        with torch.no_grad():
            context = self.context_net(position, wi, roughness)
            uv = self._direction_to_uv(wo)
            u, v = uv[:, 0:1], uv[:, 1:2]
            
            # Marginal U
            logits_u = self.marginal_head(context)
            pdf_u_vals = self._get_density(logits_u, self.df_config.resolution_u)
            val_u = self._eval_1d(pdf_u_vals, u, self.df_config.resolution_u)
            
            # Conditional V
            u_feats = self.u_encoder(u)
            cond_input = torch.cat([context, u_feats], dim=1)
            logits_v = self.conditional_net(cond_input).float()
            pdf_v_vals = self._get_density(logits_v, self.df_config.resolution_v)
            val_v = self._eval_1d(pdf_v_vals, v, self.df_config.resolution_v)
            
            pdf_uv = val_u * val_v
            pdf_dir = pdf_uv / self._jacobian()
            
            return pdf_dir.squeeze()

    def train_step(self, batch: TrainingBatch) -> float:
        if not batch.is_valid():
            return -1.0
            
        self.context_net.train()
        self.conditional_net.train()
        self.marginal_head.train()
        
        # Data
        pos = batch.position_normalized # [0,1]
        wo = batch.wo
        wi = batch.wi
        roughness = batch.roughness
        radiance = batch.radiance_rgb.mean(dim=1)
        
        # Forward
        context = self.context_net(pos, wi, roughness)
        uv = self._direction_to_uv(wo)
        u, v = uv[:, 0:1], uv[:, 1:2]
        
        # 1. Marginal
        logits_u = self.marginal_head(context)
        pdf_u_vals = self._get_density(logits_u, self.df_config.resolution_u)
        val_u = self._eval_1d(pdf_u_vals, u, self.df_config.resolution_u)
        
        # 2. Conditional
        u_feats = self.u_encoder(u)
        cond_input = torch.cat([context, u_feats], dim=1)
        logits_v = self.conditional_net(cond_input).float()
        pdf_v_vals = self._get_density(logits_v, self.df_config.resolution_v)
        val_v = self._eval_1d(pdf_v_vals, v, self.df_config.resolution_v)
        
        # Joint
        pdf_uv = val_u * val_v
        
        # With area-preserving map, jacobian is constant, 
        # so maximizing log(pdf_uv) is equivalent to maximizing log(pdf_dir)
        # up to a constant. We can train on pdf_uv directly for stability, 
        # or divide by 4pi. Let's be exact.
        pdf_dir = pdf_uv / self._jacobian()
        
        # Loss
        log_q = torch.log(pdf_dir + 1e-10)
        weight = radiance.clamp(max=10.0)
        loss = -(weight * log_q).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def sample_guided_direction(self, position, wi, roughness):
        return self.sample(position, wi, roughness)
    
    def log_pdf(self, position, wi, wo, roughness):
        return torch.log(self.pdf(position, wi, wo, roughness) + 1e-10)
    
    def train_step_from_batch(self, batch: TrainingBatch) -> float:
        return self.train_step(batch)
    
    def get_distribution_for_visualization(self, position, wi, roughness):
        self._vis_position = position
        self._vis_wi = wi
        self._vis_roughness = roughness
        return self