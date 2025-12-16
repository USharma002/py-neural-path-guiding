"""
Neural Importance Sampling (NIS) guiding distribution (paper-closer, MIS-aware training).

What this version changes vs your current nis_guiding.py:
- Uses importance weights w = radiance_scalar / combined_pdf (combined sampling density that generated wo).
- Trains against q_eff = c * p_bsdf + (1 - c) * q_guide (MIS-aware objective), where:
    c is either the logged bsdf_fraction (default, stable), or a learned selection network.
- Uses hemisphere equal-area parameterization (z in [0,1]) so guided samples never go below the surface.

This file is designed to be a drop-in replacement under your GuidingDistribution API.
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

logger = get_logger("nis_guiding")


# ======================================================================================
# Config
# ======================================================================================

@dataclass
class NISConfig(GuidingConfig):
    # Flow architecture
    num_coupling_layers: int = 4
    num_bins: int = 32
    hidden_dim: int = 128
    conditioning_dim: int = 64

    # Encoding for the conditioning net (paper-like: one-blob)
    oneblob_bins: int = 32
    use_tcnn: bool = True

    # Training
    divergence: Literal["kl", "chi2"] = "kl"
    learning_rate: float = 5e-3
    grad_clip_norm: float = 10.0
    epsilon: float = 1e-8

    # Weighting (importance weights)
    max_weight: float = 50.0
    max_radiance: float = 1e6

    # Minibatching (keep small to reduce VRAM spikes)
    batch_size: int = 65536

    # Mixture / selection
    use_logged_bsdf_fraction: bool = True  # use batch.bsdf_fraction as c (most stable with your current integrator)
    learn_selection: bool = False          # if True, learn c(x) and optimize q_eff using c_pred
    selection_hidden: int = 64


# ======================================================================================
# Hemisphere equal-area mapping: (u,v) in [0,1]^2 <-> direction with z in [0,1]
# Area element on hemisphere is dA = dphi dz, total area = 2*pi => Jacobian = 2*pi.
# ======================================================================================

def dir_to_uv_hemi(d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    d = F.normalize(d, dim=-1)
    z = d[:, 2].clamp(0.0, 1.0)  # hemisphere
    phi = torch.atan2(d[:, 1], d[:, 0])  # [-pi, pi]
    u = (phi + math.pi) / (2.0 * math.pi)
    v = z
    return torch.stack([u, v], dim=1).clamp(eps, 1.0 - eps)

def uv_to_dir_hemi(uv: torch.Tensor) -> torch.Tensor:
    u = uv[:, 0].clamp(0.0, 1.0)
    z = uv[:, 1].clamp(0.0, 1.0)
    phi = u * (2.0 * math.pi) - math.pi
    sin_theta = torch.sqrt((1.0 - z * z).clamp(min=0.0))
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    return torch.stack([x, y, z], dim=1)

def jacobian_hemi() -> float:
    return 2.0 * math.pi


# ======================================================================================
# One-blob encoding (PyTorch fallback)
# ======================================================================================

class OneBlobEncoder(nn.Module):
    def __init__(self, dims: int, bins: int):
        super().__init__()
        self.dims = dims
        self.bins = bins
        centers = torch.linspace(0.0, 1.0, bins)
        self.register_buffer("centers", centers)
        self.sigma = 1.0 / bins

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # x01: (N, D) in [0,1]
        diff = x01.unsqueeze(-1) - self.centers.view(1, 1, -1)  # (N, D, B)
        enc = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        return enc.reshape(x01.shape[0], -1)  # (N, D*B)


# ======================================================================================
# Conditioning network
# ======================================================================================

class ConditioningNetwork(nn.Module):
    """
    Paper-closer conditioning: OneBlob over (pos01, wi01, rough01) -> MLP -> conditioning vector.
    Input dims: 3 + 3 + 1 = 7.
    """

    def __init__(self, out_dim: int, oneblob_bins: int, hidden_dim: int, use_tcnn: bool):
        super().__init__()
        self.use_tcnn = bool(use_tcnn and TCNN_AVAILABLE)

        in_dims = 7
        if self.use_tcnn:
            # OneBlob encoding on all 7 dims.
            encoding = {"otype": "OneBlob", "n_bins": oneblob_bins}
            network = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 2,
            }
            self.net = tcnn.NetworkWithInputEncoding(
                n_input_dims=in_dims,
                n_output_dims=out_dim,
                encoding_config=encoding,
                network_config=network,
            )
        else:
            self.encoder = OneBlobEncoder(in_dims, oneblob_bins)
            self.net = nn.Sequential(
                nn.Linear(in_dims * oneblob_bins, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, pos01: torch.Tensor, wi01: torch.Tensor, rough01: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pos01, wi01, rough01], dim=1).clamp(0.0, 1.0)
        if self.use_tcnn:
            return self.net(x.contiguous()).float()
        return self.net(self.encoder(x))


# ======================================================================================
# Piecewise quadratic coupling (1D) conditioned on scene features
# Outputs K widths + (K+1) vertices
# ======================================================================================

class PiecewiseQuadraticCoupling(nn.Module):
    def __init__(self, num_bins: int, cond_dim: int, hidden_dim: int, oneblob_bins: int, use_tcnn: bool, eps: float):
        super().__init__()
        self.K = num_bins
        self.cond_dim = cond_dim
        self.eps = eps
        self.n_out = self.K + (self.K + 1)

        self.use_tcnn = bool(use_tcnn and TCNN_AVAILABLE)
        if self.use_tcnn:
            encoding = {
                "otype": "Composite",
                "nested": [
                    {"n_dims_to_encode": 1, "otype": "OneBlob", "n_bins": oneblob_bins},
                    {"n_dims_to_encode": cond_dim, "otype": "Identity"},
                ],
            }
            network = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 3,
            }
            self.net = tcnn.NetworkWithInputEncoding(
                n_input_dims=1 + cond_dim,
                n_output_dims=self.n_out,
                encoding_config=encoding,
                network_config=network,
            )
        else:
            self.encoder_xa = OneBlobEncoder(1, oneblob_bins)
            self.net = nn.Sequential(
                nn.Linear(oneblob_bins + cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_out),
            )

    def _params(self, xa: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # xa: (N,1) in [0,1], cond: (N,C)
        if self.use_tcnn:
            inp = torch.cat([xa, cond], dim=1).contiguous()
            out = self.net(inp).float()
        else:
            xa_enc = self.encoder_xa(xa)
            out = self.net(torch.cat([xa_enc, cond], dim=1))

        W_u, V_u = torch.split(out, [self.K, self.K + 1], dim=1)
        W = F.softmax(W_u, dim=1)

        V_u = V_u.clamp(-10, 10)
        expV = torch.exp(V_u)

        vL, vR = expV[:, :-1], expV[:, 1:]
        areas = 0.5 * (vL + vR) * W
        total = torch.sum(areas, dim=1, keepdim=True).clamp(min=self.eps)
        V = expV / total
        return W, V

    def forward(self, xb: torch.Tensor, xa: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xb = xb.clamp(0.0, 1.0 - self.eps)
        W, V = self._params(xa, cond)

        widths_cum = torch.cumsum(W, dim=1)
        pad = torch.zeros((W.shape[0], 1), device=W.device, dtype=W.dtype)
        edges = torch.cat([pad, widths_cum], dim=1)
        edges[:, -1] = 1.0

        bin_idx = torch.sum(xb >= edges[:, :-1], dim=1, keepdim=True) - 1
        bin_idx = bin_idx.clamp(0, self.K - 1)

        w_b = torch.gather(W, 1, bin_idx).clamp(min=self.eps)
        v_b = torch.gather(V, 1, bin_idx)
        v_b1 = torch.gather(V, 1, bin_idx + 1)
        edge_x = torch.gather(edges, 1, bin_idx)

        alpha = ((xb - edge_x) / w_b).clamp(0.0, 1.0)

        vL, vR = V[:, :-1], V[:, 1:]
        bin_areas = 0.5 * (vL + vR) * W
        cum_areas = torch.cumsum(bin_areas, dim=1)
        cum_areas_pad = torch.cat([pad, cum_areas], dim=1)
        area_pre = torch.gather(cum_areas_pad, 1, bin_idx)

        z = (area_pre + alpha * v_b * w_b + 0.5 * (alpha ** 2) * (v_b1 - v_b) * w_b).clamp(0.0, 1.0)

        pdf_val = (v_b + alpha * (v_b1 - v_b)).clamp(min=self.eps)
        log_det = torch.log(pdf_val)
        return z, log_det

    def inverse(self, zb: torch.Tensor, xa: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zb = zb.clamp(0.0, 1.0 - self.eps)
        W, V = self._params(xa, cond)

        widths_cum = torch.cumsum(W, dim=1)
        pad = torch.zeros((W.shape[0], 1), device=W.device, dtype=W.dtype)
        edges = torch.cat([pad, widths_cum], dim=1)
        edges[:, -1] = 1.0

        vL, vR = V[:, :-1], V[:, 1:]
        bin_areas = 0.5 * (vL + vR) * W
        cum_areas = torch.cumsum(bin_areas, dim=1)
        cum_areas_edges = torch.cat([pad, cum_areas], dim=1)
        cum_areas_edges[:, -1] = 1.0

        bin_idx = torch.sum(zb >= cum_areas_edges[:, :-1], dim=1, keepdim=True) - 1
        bin_idx = bin_idx.clamp(0, self.K - 1)

        w_b = torch.gather(W, 1, bin_idx).clamp(min=self.eps)
        v_b = torch.gather(V, 1, bin_idx)
        v_b1 = torch.gather(V, 1, bin_idx + 1)

        edge_area = torch.gather(cum_areas_edges, 1, bin_idx)
        edge_x = torch.gather(edges, 1, bin_idx)

        delta_v = v_b1 - v_b
        A = 0.5 * delta_v * w_b
        B = v_b * w_b
        C = edge_area - zb

        det = (B * B - 4.0 * A * C).clamp(min=0.0)
        sqrt_det = torch.sqrt(det)

        is_quad = torch.abs(A) > self.eps * torch.abs(B).clamp(min=1.0)
        sign_B = torch.sign(B)
        sign_B = torch.where(sign_B == 0.0, torch.ones_like(sign_B), sign_B)
        denom = (B + sign_B * sqrt_det).clamp(min=self.eps)

        alpha_q = (-2.0 * C) / denom
        alpha_l = (-C) / B.clamp(min=self.eps)
        alpha = torch.where(is_quad, alpha_q, alpha_l).clamp(0.0, 1.0)

        xb = (edge_x + alpha * w_b).clamp(0.0, 1.0)

        pdf_val = (v_b + alpha * (v_b1 - v_b)).clamp(min=self.eps)
        log_det = torch.log(pdf_val)
        return xb, log_det


# ======================================================================================
# 2D Flow on [0,1]^2
# ======================================================================================

class NISFlow(nn.Module):
    def __init__(self, cfg: NISConfig):
        super().__init__()
        self.cfg = cfg

        self.conditioning = ConditioningNetwork(
            out_dim=cfg.conditioning_dim,
            oneblob_bins=cfg.oneblob_bins,
            hidden_dim=cfg.hidden_dim,
            use_tcnn=cfg.use_tcnn,
        )

        self.layers = nn.ModuleList()
        for _ in range(cfg.num_coupling_layers):
            self.layers.append(
                PiecewiseQuadraticCoupling(
                    num_bins=cfg.num_bins,
                    cond_dim=cfg.conditioning_dim,
                    hidden_dim=cfg.hidden_dim,
                    oneblob_bins=cfg.oneblob_bins,
                    use_tcnn=cfg.use_tcnn,
                    eps=cfg.epsilon,
                )
            )

        # Alternating masks: transform dim 1 conditioned on dim 0, then swap
        masks = []
        for i in range(cfg.num_coupling_layers):
            masks.append(torch.tensor([1.0, 0.0]) if (i % 2 == 0) else torch.tensor([0.0, 1.0]))
        for i, m in enumerate(masks):
            self.register_buffer(f"mask_{i}", m)

    def forward_logprob_uv(self, uv: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = uv
        log_det_sum = torch.zeros((uv.shape[0],), device=uv.device, dtype=uv.dtype)

        for i, layer in enumerate(self.layers):
            mask = getattr(self, f"mask_{i}")

            if mask[0] == 1.0:
                xa = z[:, 0:1]
                xb = z[:, 1:2]
                dim_b = 1
            else:
                xa = z[:, 1:2]
                xb = z[:, 0:1]
                dim_b = 0

            yb, log_det = layer.forward(xb, xa, cond)

            z = z.clone()
            z[:, dim_b:dim_b + 1] = yb
            log_det_sum = log_det_sum + log_det.squeeze(-1)

        return log_det_sum  # base is uniform => log p = sum log det

    def sample_uv(self, n: int, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.rand((n, 2), device=cond.device, dtype=cond.dtype)
        log_det_sum = torch.zeros((n,), device=cond.device, dtype=cond.dtype)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            mask = getattr(self, f"mask_{i}")

            if mask[0] == 1.0:
                xa = z[:, 0:1]
                zb = z[:, 1:2]
                dim_b = 1
            else:
                xa = z[:, 1:2]
                zb = z[:, 0:1]
                dim_b = 0

            xb, log_det = layer.inverse(zb, xa, cond)

            z = z.clone()
            z[:, dim_b:dim_b + 1] = xb
            log_det_sum = log_det_sum + log_det.squeeze(-1)

        return z, log_det_sum


# ======================================================================================
# Optional selection network c(x)
# ======================================================================================

class SelectionNetwork(nn.Module):
    def __init__(self, cfg: NISConfig):
        super().__init__()
        # Uses same raw 7D input (pos01, wi01, rough01), predicts c in (0,1)
        in_dim = 7
        h = cfg.selection_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x01)).squeeze(-1)


# ======================================================================================
# Guiding distribution
# ======================================================================================

class NISGuidingDistribution(GuidingDistribution):
    def __init__(self, config: Optional[NISConfig] = None):
        cfg = config or NISConfig()
        super().__init__(cfg)
        self.cfg = cfg

        self.flow = NISFlow(cfg).to(self.device)

        self.selection = SelectionNetwork(cfg).to(self.device) if cfg.learn_selection else None

        params = list(self.flow.parameters()) + (list(self.selection.parameters()) if self.selection is not None else [])
        self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

        self._logJ = math.log(jacobian_hemi())

        logger.info(
            f"NIS initialized: L={cfg.num_coupling_layers}, K={cfg.num_bins}, "
            f"div={cfg.divergence}, learn_selection={cfg.learn_selection}, "
            f"use_logged_c={cfg.use_logged_bsdf_fraction}"
        )

    @property
    def name(self) -> str:
        return "Neural Importance Sampling (MIS-aware)"

    def _pos_to_01(self, position_ws: torch.Tensor) -> torch.Tensor:
        if self.bbox_min is None or self.bbox_max is None:
            raise RuntimeError(f"{self.name}: Scene bbox not set. Call set_scene(scene) first.")
        extent = (self.bbox_max - self.bbox_min)
        extent = torch.where(extent < 1e-6, torch.ones_like(extent), extent)
        pos01 = (position_ws - self.bbox_min) / extent
        return pos01.clamp(0.0, 1.0)

    def _wi_to_01(self, wi: torch.Tensor) -> torch.Tensor:
        wi = F.normalize(wi, dim=-1)
        return ((wi + 1.0) * 0.5).clamp(0.0, 1.0)

    def _rough_to_01(self, roughness: torch.Tensor) -> torch.Tensor:
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)
        return roughness.clamp(0.0, 1.0)

    def _cond(self, position_ws: torch.Tensor, wi: torch.Tensor, roughness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos01 = self._pos_to_01(position_ws)
        wi01 = self._wi_to_01(wi)
        r01 = self._rough_to_01(roughness)
        cond = self.flow.conditioning(pos01, wi01, r01)
        x01 = torch.cat([pos01, wi01, r01], dim=1)  # for selection net
        return cond, x01

    # ----- API -----

    def sample(self, position: torch.Tensor, wi: torch.Tensor, roughness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.flow.eval()
        with torch.no_grad():
            cond, _ = self._cond(position, wi, roughness)
            uv, logp_uv = self.flow.sample_uv(position.shape[0], cond)
            wo = uv_to_dir_hemi(uv)
            pdf = torch.exp(logp_uv - self._logJ)
            return wo, pdf

    def pdf(self, position: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        self.flow.eval()
        with torch.no_grad():
            # If wo is below hemisphere, pdf = 0
            below = wo[:, 2] <= 0.0
            cond, _ = self._cond(position, wi, roughness)
            uv = dir_to_uv_hemi(wo, eps=self.cfg.epsilon)
            logp_uv = self.flow.forward_logprob_uv(uv, cond)
            pdf = torch.exp(logp_uv - self._logJ)
            pdf = torch.where(below, torch.zeros_like(pdf), pdf)
            return pdf

    def log_pdf(self, position: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        self.flow.eval()
        with torch.no_grad():
            below = wo[:, 2] <= 0.0
            cond, _ = self._cond(position, wi, roughness)
            uv = dir_to_uv_hemi(wo, eps=self.cfg.epsilon)
            logp_uv = self.flow.forward_logprob_uv(uv, cond)
            logp = logp_uv - self._logJ
            logp = torch.where(below, torch.full_like(logp, -float("inf")), logp)
            return logp

    def selection_prob(self, position: torch.Tensor, wi: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        """Optional: learned c(x) in [0,1]. If disabled, returns constant 0.5."""
        if self.selection is None:
            return torch.full((position.shape[0],), 0.5, device=self.device)
        self.selection.eval()
        with torch.no_grad():
            _, x01 = self._cond(position, wi, roughness)
            return self.selection(x01)

    # ----- Training -----

    def train_step(self, batch: TrainingBatch) -> float:
        if not batch.is_valid():
            return -1.0

        # Required signals
        pos = batch.position
        wi = batch.wi
        wo = batch.wo
        rough = batch.roughness

        combined_pdf = batch.combined_pdf
        rad = batch.radiance_scalar

        # Optional fields (should be present in your updated TrainingBatch)
        bsdf_pdf = getattr(batch, "bsdf_pdf", None)
        bsdf_frac_logged = getattr(batch, "bsdf_fraction", None)
        is_delta = getattr(batch, "is_delta", None)
        guiding_active = getattr(batch, "guiding_active", None)

        # Basic validation
        if combined_pdf is None or rad is None:
            return -1.0

        # Filter invalid samples
        valid = torch.isfinite(pos).all(dim=1) & torch.isfinite(wi).all(dim=1) & torch.isfinite(wo).all(dim=1)
        valid &= torch.isfinite(combined_pdf) & (combined_pdf > 0)
        valid &= torch.isfinite(rad)

        if is_delta is not None and is_delta.numel() == rad.numel():
            valid &= (~is_delta.bool())  # skip delta vertices

        if guiding_active is not None and guiding_active.numel() == rad.numel():
            # optional: focus training only when guiding was active in integrator
            valid &= guiding_active.bool()

        if valid.sum().item() == 0:
            return -1.0

        pos = pos[valid]
        wi = wi[valid]
        wo = wo[valid]
        rough = rough[valid]
        combined_pdf = combined_pdf[valid]
        rad = rad[valid]

        if bsdf_pdf is not None and bsdf_pdf.numel() == batch.num_samples:
            bsdf_pdf = bsdf_pdf[valid]
        else:
            bsdf_pdf = None

        if bsdf_frac_logged is not None and bsdf_frac_logged.numel() == batch.num_samples:
            bsdf_frac_logged = bsdf_frac_logged[valid]
        else:
            bsdf_frac_logged = None

        # Random minibatch
        n = pos.shape[0]
        B = min(int(self.cfg.batch_size), n)
        idx = torch.randint(0, n, (B,), device=pos.device)

        pos = pos[idx].contiguous()
        wi = wi[idx].contiguous()
        wo = wo[idx].contiguous()
        rough = rough[idx].contiguous()
        combined_pdf = combined_pdf[idx].contiguous()
        rad = rad[idx].contiguous()
        if bsdf_pdf is not None:
            bsdf_pdf = bsdf_pdf[idx].contiguous()
        if bsdf_frac_logged is not None:
            bsdf_frac_logged = bsdf_frac_logged[idx].contiguous()

        # Importance weights: w = f / r, where r is the pdf that generated wo (combined_pdf)
        denom = combined_pdf.clamp(min=self.cfg.epsilon)
        rad = rad.clamp(min=0.0, max=self.cfg.max_radiance)
        w = (rad / denom).clamp(max=self.cfg.max_weight).detach()
        w = w / (w.mean().clamp_min(self.cfg.epsilon))  # normalize for stability

        # Compute q_guide at wo
        self.flow.train()
        if self.selection is not None:
            self.selection.train()

        cond, x01 = self._cond(pos, wi, rough)
        uv = dir_to_uv_hemi(wo, eps=self.cfg.epsilon)
        log_q_uv = self.flow.forward_logprob_uv(uv, cond)
        log_q_guide = log_q_uv - self._logJ
        q_guide = torch.exp(log_q_guide).clamp(min=self.cfg.epsilon)

        # Choose mixture coefficient c
        if self.selection is not None and self.cfg.learn_selection:
            c = self.selection(x01).clamp(self.cfg.epsilon, 1.0 - self.cfg.epsilon)
        else:
            if self.cfg.use_logged_bsdf_fraction and bsdf_frac_logged is not None:
                c = bsdf_frac_logged.clamp(0.0, 1.0)
            else:
                c = torch.full((B,), 0.5, device=self.device)

        # Form q_eff if bsdf_pdf exists; otherwise train q_guide only
        if bsdf_pdf is not None:
            p_bsdf = bsdf_pdf.clamp(min=self.cfg.epsilon)
            q_eff = (c * p_bsdf + (1.0 - c) * q_guide).clamp(min=self.cfg.epsilon)
            log_q_eff = torch.log(q_eff)
        else:
            log_q_eff = log_q_guide

        # Loss (paper-style weighted objective)
        if self.cfg.divergence == "kl":
            loss = -(w * log_q_eff).mean()
        else:
            loss = -((w * w) * log_q_eff).mean()

        if not torch.isfinite(loss):
            return -1.0

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.flow.parameters()) + (list(self.selection.parameters()) if self.selection is not None else []),
            self.cfg.grad_clip_norm
        )
        if not torch.isfinite(grad_norm):
            return -1.0

        self.optimizer.step()
        return float(loss.item())

    def get_distribution_for_visualization(self, position: torch.Tensor, wi: torch.Tensor, roughness: torch.Tensor) -> "NISGuidingDistribution":
        self._vis_position = position
        self._vis_wi = wi
        self._vis_roughness = roughness
        return self
