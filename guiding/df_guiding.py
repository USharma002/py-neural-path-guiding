"""Distribution Factorization (DF) guiding distribution.

Based on: "Neural Path Guiding with Distribution Factorization" (Figueiredo et al., 2025).
Reference: https://github.com/pedrovfigueiredo/guiding-df

Key Features:
- Factorization: p(u, v) = p(u) * p(v|u)
- Mapping: Area-preserving square-to-sphere mapping (Uniform Jacobian).
- Encodings: SH for direction, OneBlob for roughness, Triangle Wave for conditional.

All public methods accept WORLD SPACE positions and normalize internally.
Network expects positions in [0, 1]^3 range for grid encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Any
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
    
    # Conditional Encoding (Paper: Triangle Wave for epsilon_1)
    u_encoding_freqs: int = 12 
    
    # Training
    learning_rate: float = 5e-3 # Paper: 1e-2
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
        # Triangle wave: abs(2 * (x * freq % 1) - 1)
        scaled = x * self.freqs
        val = 2.0 * torch.abs(2.0 * (scaled - torch.floor(scaled + 0.5))) - 1.0
        return val


class ContextEncoder(nn.Module):
    """
    Encodes scene context (Position, Direction, Roughness).
    Paper specs:
    - Position: Learnable Dense Grid (We use HashGrid as efficient equivalent)
    - Direction: Spherical Harmonics (Degree 4)
    - Roughness: OneBlob (4 bins)
    
    Expects position in [0, 1]^3 range.
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
            self.use_tcnn = True
        else:
            # Fallback (Simpler MLP)
            self.net = nn.Sequential(
                nn.Linear(7, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, output_dim)
            )
            self.use_tcnn = False
            
    def forward(self, pos, dir, rough):
        if rough.ndim == 1: 
            rough = rough.unsqueeze(1)
        
        # Ensure inputs are in valid ranges
        pos = pos.clamp(0.0, 1.0)  # Position in [0, 1]
        dir = F.normalize(dir, dim=-1)  # Direction normalized
        rough = rough.clamp(0.0, 1.0)  # Roughness in [0, 1]
        
        x = torch.cat([pos, dir, rough], dim=1)
        
        if self.use_tcnn:
            x = x.contiguous()
            
        return self.net(x).float()


class DFGuidingDistribution(GuidingDistribution):
    """Distribution Factorization path guiding.
    
    All public methods accept WORLD SPACE positions and normalize internally.
    Network requires positions in [0, 1]^3 for grid encoding.
    """
    
    def __init__(self, config: Optional[DFConfig] = None):
        config = config or DFConfig()
        super().__init__(config)
        self.df_config = config
        
        # 1. Context Encoder
        self.context_net = ContextEncoder(
            output_dim=config.hidden_dim,
            use_tcnn=config.use_tcnn
        ).to(self.device)
        
        # 2. U-Encoding (Triangle Wave for conditional input)
        self.u_encoder = TriangleWaveEncoding(n_frequencies=config.u_encoding_freqs).to(self.device)
        u_feat_dim = config.u_encoding_freqs
        
        # 3. PDF Estimators (Heads)
        self.marginal_head = nn.Linear(config.hidden_dim, config.resolution_u).to(self.device)
        
        # Conditional Head
        cond_input_dim = config.hidden_dim + u_feat_dim
        
        if config.use_tcnn and TCNN_AVAILABLE:
            self.conditional_net = tcnn.Network(
                n_input_dims=cond_input_dim,
                n_output_dims=config.resolution_v,
                network_config={
                    "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                    "n_neurons": config.hidden_dim, "n_hidden_layers": 2
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

    def _prepare_network_input(self, position: torch.Tensor) -> torch.Tensor:
        """Convert world-space position to DF network input format [0, 1]^3.
        
        DF network expects positions normalized to [0, 1]^3 range for HashGrid encoding.
        
        Args:
            position: (N, 3) world-space positions
            
        Returns:
            (N, 3) normalized positions in [0, 1]^3
        """
        if self.bbox_min is None or self.bbox_max is None:
            raise RuntimeError(
                f"{self.name}: Scene bbox not set. Call set_scene() first."
            )
        
        # Normalize to [0, 1] using scene bbox
        extent = self.bbox_max - self.bbox_min
        extent = torch.where(extent < 1e-6, torch.ones_like(extent), extent)
        pos_01 = (position - self.bbox_min) / extent
        
        return pos_01

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
        
        # Z -> V: z = 1 - 2v  =>  v = (1 - z) / 2
        v = (1.0 - z) / 2.0
        
        return torch.stack([u, v], dim=1).clamp(0.0, 1.0)

    def _uv_to_direction(self, uv: torch.Tensor) -> torch.Tensor:
        """Inverse of above."""
        u, v = uv[:, 0], uv[:, 1]
        
        # Phi
        phi = u * (2 * torch.pi) - torch.pi
        
        # Z
        z = 1.0 - 2.0 * v
        
        # Sin theta
        sin_theta = torch.sqrt((1.0 - z**2).clamp(min=0.0))
        
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        
        return torch.stack([x, y, z], dim=1)

    def _jacobian(self) -> float:
        """
        For the area preserving mapping:
        d_omega = 4 * pi * du * dv
        Jacobian is constant 4 * pi.
        """
        return 4 * torch.pi

    # =========================================================================
    #  Sampling / Eval Utils
    # =========================================================================

    def _get_density(self, logits: torch.Tensor, M: int) -> torch.Tensor:
        """Convert logits to density via softmax."""
        return M * F.softmax(logits, dim=1)

    def _eval_1d(self, density: torch.Tensor, coords: torch.Tensor, M: int, is_periodic: bool = False) -> torch.Tensor:
        """
        Evaluates the PDF at specific coordinates [0, 1].
        Handles periodic (U) and clamped (V) boundaries.
        """
        if self.df_config.interpolation_mode == "nearest":
            idx = (coords * M).long().clamp(0, M - 1)
            return torch.gather(density, 1, idx)
        else:
            # Linear interpolation
            m = coords * M
            idx_floor = torch.floor(m).long()

            if is_periodic:
                # Wrap indices for periodic dimension (U/phi)
                idx_l = idx_floor % M
                idx_r = (idx_floor + 1) % M
            else:
                # Clamp indices for non-periodic dimension (V/theta)
                idx_l = idx_floor.clamp(0, M - 1)
                idx_r = (idx_floor + 1).clamp(0, M - 1)
            
            # Local coordinate
            alpha = (m - idx_floor).clamp(0, 1)
            
            val_l = torch.gather(density, 1, idx_l)
            val_r = torch.gather(density, 1, idx_r)
            
            return (1 - alpha) * val_l + alpha * val_r

    def _sample_1d(self, density: torch.Tensor, rand: torch.Tensor, M: int, is_periodic: bool = False) -> torch.Tensor:
        """Inverse Transform Sampling for Piecewise Linear distributions."""
        B = density.shape[0]
        
        # 1. Construct Vertex Pairs
        v_curr = density
        if is_periodic:
            v_next = torch.cat([density[:, 1:], density[:, 0:1]], dim=1)
        else:
            v_next = torch.cat([density[:, 1:], density[:, -1:]], dim=1)
            
        # 2. Build CDF
        bin_width = 1.0 / M
        areas = 0.5 * (v_curr + v_next) * bin_width
        cdf = torch.cumsum(areas, dim=1)
        
        total_area = cdf[:, -1:]
        cdf = cdf / (total_area + 1e-8)
        
        cdf_full = torch.cat([torch.zeros(B, 1, device=self.device), cdf], dim=1)
        
        # 3. Find Bin Index
        idx = torch.searchsorted(cdf_full, rand) - 1
        idx = idx.clamp(0, M - 1)
        
        # 4. Invert Quadratic
        cdf_lo = torch.gather(cdf_full, 1, idx)
        target_norm = rand - cdf_lo
        bin_area_norm = torch.gather(cdf_full, 1, idx+1) - cdf_lo
        
        frac = target_norm / (bin_area_norm + 1e-10)
        
        y0 = torch.gather(v_curr, 1, idx)
        y1 = torch.gather(v_next, 1, idx)
        delta = y1 - y0
        
        A = 0.5 * delta
        B_coef = y0
        C = -frac * (y0 + 0.5 * delta)
        
        det = (B_coef**2 - 4*A*C).clamp(min=0)
        sqrt_det = torch.sqrt(det)
        
        t_quad = (-B_coef + sqrt_det) / (2*A + 1e-10)
        t_lin = -C / (B_coef + 1e-10)
        
        is_lin = torch.abs(A) < 1e-5
        t = torch.where(is_lin, t_lin, t_quad).clamp(0, 1)
        
        return (idx + t) / M

    # =========================================================================
    #  Main Interface
    # =========================================================================

    def sample(self, position, wi, roughness):
        """Sample directions from the DF distribution.
        
        Args:
            position: (N, 3) positions in WORLD SPACE
            wi: (N, 3) incoming directions
            roughness: (N, 1) surface roughness
            
        Returns:
            Tuple of:
            - directions: (N, 3) sampled unit vectors
            - pdf_values: (N,) probability densities
        """
        self.context_net.eval()
        self.conditional_net.eval()
        self.marginal_head.eval()
        
        B = position.shape[0]
        
        # Normalize world space -> [0, 1] for DF network
        pos_normalized = self._prepare_network_input(position)
        
        with torch.no_grad():
            # 1. Context Features
            context = self.context_net(pos_normalized, wi, roughness)
            
            # 2. Sample U (Marginal) - Periodic (Phi)
            logits_u = self.marginal_head(context)
            pdf_u_vals = self._get_density(logits_u, self.df_config.resolution_u)
            
            rand_u = torch.rand(B, 1, device=self.device)
            u_samples = self._sample_1d(pdf_u_vals, rand_u, self.df_config.resolution_u, is_periodic=True)
            
            # 3. Sample V (Conditional) - Clamped (Theta)
            u_feats = self.u_encoder(u_samples)
            cond_input = torch.cat([context, u_feats], dim=1)
            
            logits_v = self.conditional_net(cond_input).float()
            pdf_v_vals = self._get_density(logits_v, self.df_config.resolution_v)
            
            rand_v = torch.rand(B, 1, device=self.device)
            v_samples = self._sample_1d(pdf_v_vals, rand_v, self.df_config.resolution_v, is_periodic=False)
            
            # 4. Combine
            uv = torch.cat([u_samples, v_samples], dim=1)
            directions = self._uv_to_direction(uv)
            
            # 5. PDF Calculation
            val_u = self._eval_1d(pdf_u_vals, u_samples, self.df_config.resolution_u, is_periodic=True)
            val_v = self._eval_1d(pdf_v_vals, v_samples, self.df_config.resolution_v, is_periodic=False)
            
            pdf_uv = val_u * val_v
            pdf_dir = pdf_uv / self._jacobian()
            
            return directions, pdf_dir.squeeze()

    def pdf(self, position, wi, wo, roughness):
        """Evaluate PDF for given directions.
        
        Args:
            position: (N, 3) positions in WORLD SPACE
            wi: (N, 3) incoming directions
            wo: (N, 3) outgoing directions to evaluate
            roughness: (N, 1) surface roughness
            
        Returns:
            (N,) probability density values
        """
        self.context_net.eval()
        self.conditional_net.eval()
        self.marginal_head.eval()
        
        # Normalize world space -> [0, 1] for DF network
        pos_normalized = self._prepare_network_input(position)
        
        with torch.no_grad():
            context = self.context_net(pos_normalized, wi, roughness)
            uv = self._direction_to_uv(wo)
            u, v = uv[:, 0:1], uv[:, 1:2]
            
            # Marginal U
            logits_u = self.marginal_head(context)
            pdf_u_vals = self._get_density(logits_u, self.df_config.resolution_u)
            val_u = self._eval_1d(pdf_u_vals, u, self.df_config.resolution_u, is_periodic=True)
            
            # Conditional V
            u_feats = self.u_encoder(u)
            cond_input = torch.cat([context, u_feats], dim=1)
            logits_v = self.conditional_net(cond_input).float()
            pdf_v_vals = self._get_density(logits_v, self.df_config.resolution_v)
            val_v = self._eval_1d(pdf_v_vals, v, self.df_config.resolution_v, is_periodic=False)
            
            pdf_uv = val_u * val_v
            pdf_dir = pdf_uv / self._jacobian()
            
            return pdf_dir.squeeze()

    def log_pdf(self, position, wi, wo, roughness):
        """Evaluate log PDF for given directions.
        
        Args:
            position: (N, 3) positions in WORLD SPACE
            wi: (N, 3) incoming directions
            wo: (N, 3) outgoing directions to evaluate
            roughness: (N, 1) surface roughness
            
        Returns:
            (N,) log probability density values
        """
        return torch.log(self.pdf(position, wi, wo, roughness) + 1e-10)

    def train_step(self, batch: TrainingBatch) -> float:
        if not batch.is_valid(): 
            return -1.0
        
        self.context_net.train()
        self.conditional_net.train()
        self.marginal_head.train()
        
        # Normalize world space positions to [0, 1]
        pos_normalized = self._prepare_network_input(batch.position)
        
        all_wo = batch.wo
        all_wi = batch.wi
        all_roughness = batch.roughness
        # Clamp radiance harder to prevent spikes
        all_radiance = batch.radiance_rgb.mean(dim=1).clamp(max=20.0) 
        
        total_samples = pos_normalized.shape[0]
        BATCH_SIZE = 2 * 1024 
        indices = torch.randperm(total_samples, device=self.device)
        total_loss = 0.0
        num_batches = 0
        
        jacobian_val = self._jacobian() # 4 * pi
        log_jacobian = math.log(jacobian_val)

        for start_idx in range(0, total_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_samples)
            batch_idx = indices[start_idx:end_idx]
            
            pos = pos_normalized[batch_idx].contiguous()
            wo = all_wo[batch_idx].contiguous()
            wi = all_wi[batch_idx].contiguous()
            roughness = all_roughness[batch_idx].contiguous()
            radiance = all_radiance[batch_idx].contiguous().detach()

            # --- Forward Pass ---
            context = self.context_net(pos, wi, roughness)
            uv = self._direction_to_uv(wo)
            u, v = uv[:, 0:1], uv[:, 1:2]
            
            # 1. Marginal U (Log-Space)
            logits_u = self.marginal_head(context)
            log_prob_u_grid = F.log_softmax(logits_u, dim=1) + math.log(self.df_config.resolution_u)
            prob_u_grid = torch.exp(log_prob_u_grid)
            val_u = self._eval_1d(prob_u_grid, u, self.df_config.resolution_u, is_periodic=True)
            
            # 2. Conditional V (Log-Space)
            u_feats = self.u_encoder(u)
            cond_input = torch.cat([context, u_feats], dim=1)
            logits_v = self.conditional_net(cond_input).float()
            
            log_prob_v_grid = F.log_softmax(logits_v, dim=1) + math.log(self.df_config.resolution_v)
            prob_v_grid = torch.exp(log_prob_v_grid)
            val_v = self._eval_1d(prob_v_grid, v, self.df_config.resolution_v, is_periodic=False)
            
            # 3. Joint Log Probability
            # Use log product rule: log(a * b) = log(a) + log(b)
            # This is safer than multiplying small probabilities then logging
            prob_uv = (val_u * val_v).clamp(min=1e-6) # Clamp to avoid log(0)
            log_prob_uv = torch.log(prob_uv)
            
            # Correct for Jacobian: p(w) = p(uv) / 4pi => log p(w) = log p(uv) - log(4pi)
            log_prob_dir = log_prob_uv - log_jacobian
            
            # 4. Loss
            loss = -(radiance * log_prob_dir).mean()
            
            # --- Optimization with Safety Checks ---
            self.optimizer.zero_grad()
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN Loss detected! Skipping batch. (Radiance Max: {radiance.max()})")
                continue # Skip this bad batch
                
            loss.backward()
            
            # Clip Gradients globally (all parameters)
            all_params = (
                list(self.context_net.parameters()) + 
                list(self.marginal_head.parameters()) + 
                list(self.conditional_net.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / max(1, num_batches)

    def get_distribution_for_visualization(self, position, wi, roughness):
        """Return self for visualization - visualization uses pdf() directly.
        
        Args:
            position: (1, 3) position in WORLD SPACE
            wi: (1, 3) incoming direction
            roughness: (1, 1) surface roughness
            
        Returns:
            Self (visualization calls pdf() with stored context)
        """
        self._vis_position = position
        self._vis_wi = wi
        self._vis_roughness = roughness
        return self