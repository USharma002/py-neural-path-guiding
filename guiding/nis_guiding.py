"""Neural Importance Sampling (Muller et al. 2019): piecewise-quadratic coupling flow over directions.

Learns q proportional to the product f * L_i * |cos| with MIS-aware optimization of
q' = c * p_bsdf + (1 - c) * q, where c is the integrator's BSDF selection probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import math
import mitsuba as mi
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from guiding.base import GuidingDistribution, GuidingConfig
from guiding.training_data import TrainingBatch
from guiding.config import get_logger
from integrators.path_guiding_integrator import PathGuidingIntegrator
from utils.math_utils import dir_to_square, square_to_dir, to_unit

logger = get_logger("nis_guiding")


@dataclass
class NISTrainingBatch(TrainingBatch):
    bsdf_pdf: torch.Tensor          # (N,)
    bsdf_fraction: torch.Tensor     # (N,) BSDF selection probability c of q'
    is_delta: torch.Tensor          # (N,) bool


@dataclass
class NISConfig(GuidingConfig):
    num_coupling_layers: int = 4
    num_bins: int = 32
    hidden_dim: int = 128
    conditioning_dim: int = 64
    oneblob_bins: int = 32

    divergence: Literal["kl", "chi2"] = "kl"
    learning_rate: float = 5e-3
    chi2_grad_clip_norm: float = 50.0
    epsilon: float = 1e-8


def _mlp(n_neurons: int, n_hidden_layers: int) -> dict:
    return {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
            "n_neurons": n_neurons, "n_hidden_layers": n_hidden_layers}


class PiecewiseQuadraticCoupling(nn.Module):
    """1D piecewise-quadratic warp of xb conditioned on (xa, cond); predicts K widths and K+1 vertices."""

    def __init__(self, num_bins: int, cond_dim: int, hidden_dim: int, oneblob_bins: int, eps: float):
        super().__init__()
        self.K = num_bins
        self.eps = eps
        encoding = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 1, "otype": "OneBlob", "n_bins": oneblob_bins},
                {"n_dims_to_encode": cond_dim, "otype": "Identity"},
            ],
        }
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=1 + cond_dim,
            n_output_dims=2 * self.K + 1,
            encoding_config=encoding,
            network_config=_mlp(hidden_dim, 3),
        )

    def _params(self, xa: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([xa, cond], dim=1).contiguous()).float()

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
        return z, torch.log(pdf_val)

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
        return xb, torch.log(pdf_val)


class NISFlow(nn.Module):
    """Coupling flow on [0,1]^2 with a uniform base; layers alternate which dimension they warp."""

    def __init__(self, cfg: NISConfig):
        super().__init__()
        # One-blob encoded (position, wi, normal, roughness) -> conditioning vector shared by all layers
        self.conditioning = tcnn.NetworkWithInputEncoding(
            n_input_dims=10,
            n_output_dims=cfg.conditioning_dim,
            encoding_config={"otype": "OneBlob", "n_bins": cfg.oneblob_bins},
            network_config=_mlp(cfg.hidden_dim, 2),
        )
        self.layers = nn.ModuleList(
            PiecewiseQuadraticCoupling(cfg.num_bins, cfg.conditioning_dim, cfg.hidden_dim, cfg.oneblob_bins, cfg.epsilon)
            for _ in range(cfg.num_coupling_layers)
        )

    @staticmethod
    def _split(i: int) -> Tuple[int, int]:
        """(conditioning dim, transformed dim) of layer i."""
        return (0, 1) if i % 2 == 0 else (1, 0)

    def forward_logprob_uv(self, uv: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = uv
        log_det_sum = torch.zeros((uv.shape[0],), device=uv.device, dtype=uv.dtype)
        for i, layer in enumerate(self.layers):
            a, b = self._split(i)
            yb, log_det = layer.forward(z[:, b:b + 1], z[:, a:a + 1], cond)
            z = z.clone()
            z[:, b:b + 1] = yb
            log_det_sum = log_det_sum + log_det.squeeze(-1)
        return log_det_sum

    def sample_uv(self, n: int, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.rand((n, 2), device=cond.device, dtype=cond.dtype)
        log_det_sum = torch.zeros((n,), device=cond.device, dtype=cond.dtype)
        for i in reversed(range(len(self.layers))):
            a, b = self._split(i)
            xb, log_det = self.layers[i].inverse(z[:, b:b + 1], z[:, a:a + 1], cond)
            z = z.clone()
            z[:, b:b + 1] = xb
            log_det_sum = log_det_sum + log_det.squeeze(-1)
        return z, log_det_sum


class NISGuidingDistribution(GuidingDistribution):
    def __init__(self, config: Optional[NISConfig] = None):
        cfg = config or NISConfig()
        super().__init__(cfg)
        self.cfg = cfg

        self.flow = NISFlow(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=cfg.learning_rate)
        self._log_jacobian = math.log(4 * math.pi)

        logger.info(f"NIS initialized: L={cfg.num_coupling_layers}, K={cfg.num_bins}, div={cfg.divergence}")

    @property
    def name(self) -> str:
        return "Neural Importance Sampling"

    def _cond(self, position, wi, normal, roughness) -> torch.Tensor:
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)
        x = torch.cat([
            self._position_to_unit_cube(position), to_unit(wi), to_unit(normal), roughness,
        ], dim=1).clamp(0.0, 1.0)
        return self.flow.conditioning(x.contiguous()).float()

    def _log_pdf(self, cond: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        uv = dir_to_square(F.normalize(wo, dim=-1))
        return self.flow.forward_logprob_uv(uv, cond) - self._log_jacobian

    def sample(self, position, wi, normal, roughness) -> Tuple[torch.Tensor, torch.Tensor]:
        self.flow.eval()
        with torch.no_grad():
            cond = self._cond(position, wi, normal, roughness)
            uv, logp_uv = self.flow.sample_uv(position.shape[0], cond)
            return square_to_dir(uv), torch.exp(logp_uv - self._log_jacobian)

    def pdf(self, position, wi, normal, roughness, wo) -> torch.Tensor:
        self.flow.eval()
        with torch.no_grad():
            return torch.exp(self._log_pdf(self._cond(position, wi, normal, roughness), wo))

    def train_step(self, batch: NISTrainingBatch) -> float:
        if not batch.is_valid():
            return -1.0

        g = batch.guide_weight_rgb
        F_over_r = 0.2126 * g[:, 0] + 0.7152 * g[:, 1] + 0.0722 * g[:, 2]  # F / r, r = combined_pdf

        valid = (torch.isfinite(F_over_r) & torch.isfinite(batch.combined_pdf) & (batch.combined_pdf > 0)
                 & ~batch.is_delta)
        if not valid.any():
            return -1.0
        sel = lambda t: t[valid].contiguous()

        self.flow.train()
        cond = self._cond(sel(batch.position), sel(batch.wi), sel(batch.normal), sel(batch.roughness))
        q = torch.exp(self._log_pdf(cond, sel(batch.wo)))

        c = sel(batch.bsdf_fraction)
        q_mix = c * sel(batch.bsdf_pdf) + (1.0 - c) * q
        log_q_mix = torch.log(q_mix.clamp(min=self.cfg.epsilon))

        # Gradient estimators for samples drawn from r (Muller et al. 2019, Sec. 5):
        #   KL:  -(F / r) * grad log q'      chi^2:  -(F / r) * (F / q') * grad log q'
        w = sel(F_over_r)
        if self.cfg.divergence == "chi2":
            w = w * w * sel(batch.combined_pdf) / q_mix
        loss = -(w.detach() * log_q_mix).mean()

        if not torch.isfinite(loss):
            return -1.0

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.divergence == "chi2":
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.cfg.chi2_grad_clip_norm)
        self.optimizer.step()
        return float(loss.item())


class NISIntegrator(PathGuidingIntegrator):
    distribution_class = NISGuidingDistribution
    config_class = NISConfig
    EXTRA_RECORD_FIELDS = {"bsdfPdf": mi.Float, "isDelta": mi.Bool}
    batch_class = NISTrainingBatch

    def batch_extras(self, field, n: int) -> Dict[str, torch.Tensor]:
        return dict(
            bsdf_pdf=field("bsdfPdf"),
            bsdf_fraction=torch.full((n,), float(self.bsdfSamplingFraction), device="cuda"),
            is_delta=field("isDelta").to(torch.bool),
        )
