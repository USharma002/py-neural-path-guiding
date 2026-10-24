"""Distribution Factorization guiding (Figueiredo et al. 2025, https://github.com/pedrovfigueiredo/guiding-df).

p(u, v) = p(u) * p(v | u) on an area-preserving square-to-sphere mapping, each factor a 1D grid
of M values (DF-N: nearest, DF-L: linear interpolation). The target (Eq. 14) uses a radiance
cache, here the integrator's NRC: p_hat(w) = f * cos * L_cache(x', w') / L_cache(x, wi).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple
import math

import drjit as dr
import mitsuba as mi
import tinycudann as tcnn
import torch
import torch.nn.functional as F

from guiding.base import GuidingDistribution, GuidingConfig
from guiding.training_data import TrainingBatch
from guiding.config import get_logger
from integrators.path_guiding_integrator import PathGuidingIntegrator
from utils.math_utils import dir_to_square, square_to_dir, to_unit

logger = get_logger("df_guiding")


@dataclass
class DFConfig(GuidingConfig):
    interpolation_mode: Literal["nearest", "linear"] = "linear"  # DF-N / DF-L
    resolution_u: int = 32   # M1 (azimuth, periodic)
    resolution_v: int = 16   # M2 (polar)
    learning_rate: float = 3e-2


def _context_encodings() -> list:
    """Position: dense grid (as in NPM); wi: SH degree 4; normal and roughness: one-blob, 4 bins."""
    return [
        {"n_dims_to_encode": 3, "otype": "DenseGrid", "n_levels": 8, "n_features_per_level": 4,
         "base_resolution": 8, "per_level_scale": (86 / 8) ** (1 / 7)},
        {"n_dims_to_encode": 3, "otype": "SphericalHarmonics", "degree": 4},
        {"n_dims_to_encode": 4, "otype": "OneBlob", "n_bins": 4},
    ]


def _network(n_in: int, n_out: int, encodings: list) -> tcnn.NetworkWithInputEncoding:
    return tcnn.NetworkWithInputEncoding(
        n_input_dims=n_in, n_output_dims=n_out,
        encoding_config={"otype": "Composite", "nested": encodings},
        network_config={"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                        "n_neurons": 64, "n_hidden_layers": 3},
    )


class DFGuidingDistribution(GuidingDistribution):
    def __init__(self, config: Optional[DFConfig] = None):
        config = config or DFConfig()
        super().__init__(config)
        self.df_config = config

        self.marginal_net = _network(10, config.resolution_u, _context_encodings()).to(self.device)
        self.conditional_net = _network(
            11, config.resolution_v,
            _context_encodings() + [{"n_dims_to_encode": 1, "otype": "TriangleWave", "n_frequencies": 12}],
        ).to(self.device)

        params = list(self.marginal_net.parameters()) + list(self.conditional_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        self._log_jacobian = math.log(4 * math.pi)

        logger.info(f"DF Guiding ({config.interpolation_mode.upper()}) initialized. "
                    f"Res: {config.resolution_u}x{config.resolution_v}")

    @property
    def name(self) -> str:
        return f"DF-{self.df_config.interpolation_mode[0].upper()}"

    # --- 1D grid densities on [0, 1] ---

    @staticmethod
    def _density(logits: torch.Tensor) -> torch.Tensor:
        """M grid values summing to M (softmax * M); both interpolation modes then integrate to 1."""
        return logits.shape[1] * F.softmax(logits.float(), dim=1)

    def _eval_1d(self, d: torch.Tensor, x: torch.Tensor, periodic: bool) -> torch.Tensor:
        """Density at x (B, 1). Linear: values sit at bin centers (m = xM - 0.5); u wraps, v is nearest at the ends."""
        M = d.shape[1]
        if self.df_config.interpolation_mode == "nearest":
            return torch.gather(d, 1, (x * M).long().clamp(0, M - 1))
        m = x * M - 0.5
        i0 = torch.floor(m).long()
        alpha = m - i0
        if periodic:
            i0, i1 = i0 % M, (i0 + 1) % M
        else:
            i0, i1 = i0.clamp(0, M - 1), (i0 + 1).clamp(0, M - 1)
        return (1 - alpha) * torch.gather(d, 1, i0) + alpha * torch.gather(d, 1, i1)

    def _segments(self, d: torch.Tensor, periodic: bool) -> Tuple[torch.Tensor, ...]:
        """The density as linear segments: start x0 (S,), width (S,), end values y0, y1 (B, S)."""
        B, M = d.shape
        dev = d.device
        if self.df_config.interpolation_mode == "nearest":
            return torch.arange(M, device=dev) / M, torch.full((M,), 1.0 / M, device=dev), d, d
        if periodic:
            left = right = 0.5 * (d[:, -1:] + d[:, :1])
        else:
            left, right = d[:, :1], d[:, -1:]
        x0 = torch.cat([torch.zeros(1, device=dev), (torch.arange(M, device=dev) + 0.5) / M])
        width = torch.cat([torch.full((1,), 0.5 / M, device=dev), torch.full((M - 1,), 1.0 / M, device=dev),
                           torch.full((1,), 0.5 / M, device=dev)])
        return x0, width, torch.cat([left, d], dim=1), torch.cat([d, right], dim=1)

    def _sample_1d(self, d: torch.Tensor, rand: torch.Tensor, periodic: bool) -> torch.Tensor:
        """Inverse-CDF sampling of the piecewise-linear (or constant) density; rand is (B, 1)."""
        x0, width, y0, y1 = self._segments(d, periodic)
        areas = 0.5 * (y0 + y1) * width
        cdf = torch.cumsum(areas, dim=1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf / cdf[:, -1:]], dim=1)

        idx = (torch.searchsorted(cdf, rand) - 1).clamp(0, areas.shape[1] - 1)
        cdf_lo = torch.gather(cdf, 1, idx)
        frac = (rand - cdf_lo) / (torch.gather(cdf, 1, idx + 1) - cdf_lo + 1e-10)

        # Solve 0.5 (b - a) t^2 + a t = frac (a + b) / 2 for the in-segment offset t
        a, b = torch.gather(y0, 1, idx), torch.gather(y1, 1, idx)
        A, Bq, C = 0.5 * (b - a), a, -frac * 0.5 * (a + b)
        sqrt_det = torch.sqrt((Bq * Bq - 4 * A * C).clamp(min=0))
        t_quad = (-Bq + sqrt_det) / (2 * A + 1e-10)
        t_lin = -C / (Bq + 1e-10)
        # t < 1 keeps the sample inside its segment, so _eval_1d reads the same bin
        t = torch.where(torch.abs(A) < 1e-5, t_lin, t_quad).clamp(0, 1 - 1e-6)
        return x0[idx] + t * width[idx]

    # --- Networks ---

    def _context(self, position, wi, normal, roughness) -> torch.Tensor:
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)
        return torch.cat([self._position_to_unit_cube(position),
                          to_unit(wi), to_unit(normal), roughness.clamp(0.0, 1.0)], dim=1)

    def _densities_u(self, ctx: torch.Tensor) -> torch.Tensor:
        return self._density(self.marginal_net(ctx.contiguous()))

    def _densities_v(self, ctx: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self._density(self.conditional_net(torch.cat([ctx, u], dim=1).contiguous()))

    def _log_pdf(self, ctx: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        uv = dir_to_square(F.normalize(wo, dim=-1))
        u, v = uv[:, 0:1], uv[:, 1:2]
        p_u = self._eval_1d(self._densities_u(ctx), u, periodic=True)
        p_v = self._eval_1d(self._densities_v(ctx, u), v, periodic=False)
        return (torch.log(p_u * p_v) - self._log_jacobian).squeeze(1)

    def sample(self, position, wi, normal, roughness):
        with torch.no_grad():
            ctx = self._context(position, wi, normal, roughness)
            B = ctx.shape[0]
            d_u = self._densities_u(ctx)
            u = self._sample_1d(d_u, torch.rand(B, 1, device=self.device), periodic=True)
            d_v = self._densities_v(ctx, u)
            v = self._sample_1d(d_v, torch.rand(B, 1, device=self.device), periodic=False)

            p_uv = self._eval_1d(d_u, u, periodic=True) * self._eval_1d(d_v, v, periodic=False)
            return square_to_dir(torch.cat([u, v], dim=1)), (p_uv / (4 * math.pi)).squeeze(1)

    def pdf(self, position, wi, normal, roughness, wo):
        with torch.no_grad():
            return torch.exp(self._log_pdf(self._context(position, wi, normal, roughness), wo))

    def train_step(self, batch: TrainingBatch) -> float:
        """KL gradient with Monte Carlo weights p_hat / q (Eq. 12); batch.guide_weight_rgb holds p_hat / q."""
        if not batch.is_valid():
            return -1.0

        weight = batch.guide_weight_rgb.mean(dim=1)
        valid = torch.isfinite(weight)
        sel = lambda t: t[valid]

        ctx = self._context(sel(batch.position), sel(batch.wi), sel(batch.normal), sel(batch.roughness))
        loss = -(sel(weight).detach() * self._log_pdf(ctx, sel(batch.wo))).mean()

        if not torch.isfinite(loss):
            logger.warning("Non-finite DF loss; skipping step")
            return -1.0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


@dataclass
class DFTrainingBatch(TrainingBatch):
    bsdf_weight: torch.Tensor       # (N, 3) f * cos / p_sample
    next_position: torch.Tensor     # (N, 3) x' hit along wo
    next_wi: torch.Tensor           # (N, 3) direction back toward x, local at x'
    next_normal: torch.Tensor       # (N, 3)
    has_next: torch.Tensor          # (N,) bool: wo hit a surface that was recorded


class DFIntegrator(PathGuidingIntegrator):
    distribution_class = DFGuidingDistribution
    config_class = DFConfig
    BSDF_SAMPLING_FRACTION = 0.3          # the paper guides 70% of paths
    TRAINING_BUDGET_PERCENT = 30          # and trains during the first 30% of the budget
    CACHE_MINIBATCHES = 4                 # cache trained on >= 4 minibatches before the pdf step
    EXTRA_RECORD_FIELDS = {"next_position": mi.Vector3f, "next_wi": mi.Vector3f,
                           "next_normal": mi.Vector3f, "has_next": mi.Bool}
    batch_class = DFTrainingBatch

    def record_extras(self, vertex, index, active) -> None:
        pass  # next-vertex fields are filled after tracing, in process_incoming_radiance

    def process_incoming_radiance(self) -> None:
        super().process_incoming_radiance()
        rec = self.surfaceInteractionRecord
        idx = dr.arange(mi.UInt32, self.array_size)
        has_slot = rec.active & (idx % self.max_depth != self.max_depth - 1)
        nxt = idx + 1
        rec.has_next = has_slot & dr.gather(mi.Bool, rec.active, nxt, has_slot)
        rec.next_position = dr.gather(mi.Vector3f, rec.position, nxt, rec.has_next)
        rec.next_wi = dr.gather(mi.Vector3f, rec.wi, nxt, rec.has_next)
        rec.next_normal = dr.gather(mi.Vector3f, rec.normal, nxt, rec.has_next)

    def batch_extras(self, field, n: int) -> Dict[str, torch.Tensor]:
        return dict(bsdf_weight=field("bsdf"), next_position=field("next_position"), next_wi=field("next_wi"),
                    next_normal=field("next_normal"), has_next=field("has_next").to(torch.bool))

    def train_step(self, batch: DFTrainingBatch, train_guide: bool = True, train_nrc: bool = False):
        """The cache is part of DF, so it is trained whenever the pdf is, then provides the pdf's target."""
        if not batch.is_valid():
            return (-1.0 if train_guide else None), (-1.0 if train_nrc else None)

        nrc_loss = None
        if train_guide or train_nrc:
            chunks = torch.randperm(batch.num_samples, device=batch.position.device).chunk(self.CACHE_MINIBATCHES)
            nrc_loss = sum(self.nrc_system.train_step_from_batch(batch.subset(c)) for c in chunks) / len(chunks)

        guide_loss = None
        if train_guide:
            with torch.no_grad():
                L_next = self.nrc_system.query(batch.next_position, batch.next_normal, batch.next_wi, batch.roughness)
                L_next = L_next * batch.has_next.unsqueeze(1)
                L_here = self.nrc_system.query(batch.position, batch.normal, batch.wi, batch.roughness)
                p_hat_over_q = batch.bsdf_weight * L_next / L_here.mean(dim=1, keepdim=True)
            guide_loss = self.guide.train_step(dataclasses.replace(batch, guide_weight_rgb=p_hat_over_q))

        return guide_loss, nrc_loss
