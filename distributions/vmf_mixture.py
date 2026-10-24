"""Batched von Mises-Fisher mixture."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from utils.math_utils import M_EPSILON, spherical_to_cartesian


def vmf_sample_spherical(
    kappas: torch.Tensor,
    thetas: torch.Tensor,
    phis: torch.Tensor
) -> torch.Tensor:
    """Sample one direction per vMF lobe (Wood 1994 / Jakob 2012); inputs are (B,)."""
    device = kappas.device
    batch_size = kappas.shape[0]

    mu = spherical_to_cartesian(thetas, phis)

    kappa_zero_mask = kappas < 1e-5
    u = torch.rand(batch_size, device=device)

    w_uniform = 2 * u - 1
    w_vmf = 1 + (1 / kappas) * torch.log(u + (1 - u) * torch.exp(-2 * kappas))

    w = torch.where(kappa_zero_mask, w_uniform, w_vmf)
    w = torch.clamp(w, -1.0 + M_EPSILON, 1.0 - M_EPSILON)

    s = torch.copysign(torch.ones_like(mu[:, 2]), mu[:, 2])
    a = -1.0 / (s + mu[:, 2])
    b = mu[:, 0] * mu[:, 1] * a

    v1 = torch.stack([1.0 + s * mu[:, 0]**2 * a, s * b, -s * mu[:, 0]], dim=1)
    v2 = torch.stack([b, s + mu[:, 1]**2 * a, -mu[:, 1]], dim=1)

    u_phi = torch.rand(batch_size, device=device)
    azimuthal_angle = 2 * math.pi * u_phi

    sin_theta = torch.sqrt(1 - w**2)
    samples = (
        sin_theta.unsqueeze(1) * (
            v1 * torch.cos(azimuthal_angle).unsqueeze(1) +
            v2 * torch.sin(azimuthal_angle).unsqueeze(1)
        ) +
        w.unsqueeze(1) * mu
    )

    return samples


class BatchedMixedSphericalGaussianDistribution(nn.Module):
    """Mixture of K vMF lobes per batch row; params are (B, K) tensors from GuidingNetwork."""

    def __init__(self, network_params: Dict[str, torch.Tensor]):
        super().__init__()
        self.weights = network_params['lambda']   # softmax output, rows sum to 1
        self.kappas = network_params['kappa']
        self.thetas = network_params['theta']
        self.phis = network_params['phi']

    def pdf(self, wi: torch.Tensor) -> torch.Tensor:
        """Mixture pdf of directions (B, 3) -> (B,)."""
        mus = spherical_to_cartesian(self.thetas, self.phis)
        cos = torch.sum(mus * wi.unsqueeze(1), dim=2)
        # vMF: k / (2pi (1 - e^{-2k})) * e^{k (cos - 1)}; expm1 keeps the k -> 0 limit 1 / (4pi)
        norm = self.kappas / (2 * math.pi * -torch.expm1(-2 * self.kappas))
        return torch.sum(self.weights * norm * torch.exp(self.kappas * (cos - 1)), dim=1)

    def sample(self) -> torch.Tensor:
        """One direction per batch row, (B, 3)."""
        idx = torch.multinomial(self.weights, num_samples=1)
        pick = lambda t: torch.gather(t, 1, idx).squeeze(1)
        return vmf_sample_spherical(pick(self.kappas), pick(self.thetas), pick(self.phis))
