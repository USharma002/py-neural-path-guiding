"""Neural Parametric Mixtures (Dong et al. 2023): network-predicted mixture of von Mises-Fisher lobes.

Product variant: the mixture is conditioned on wi and learns f * L_i * |cos|, trained with the
Monte Carlo KL gradient (Eq. 9): -(F / p_sample) * grad log V.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch

from guiding.base import GuidingDistribution, GuidingConfig
from networks.guiding_network import GuidingNetwork
from distributions.vmf_mixture import BatchedMixedSphericalGaussianDistribution
from guiding.training_data import TrainingBatch
from guiding.config import get_logger
from integrators.path_guiding_integrator import PathGuidingIntegrator
from utils.math_utils import to_unit

logger = get_logger("vmf_guiding")


@dataclass
class VMFConfig(GuidingConfig):
    num_lobes: int = 8
    learning_rate: float = 5e-3


class VMFGuidingDistribution(GuidingDistribution):
    def __init__(self, config: Optional[VMFConfig] = None):
        config = config or VMFConfig()
        super().__init__(config)
        self.vmf_config = config

        self.network = GuidingNetwork(device=self.device, K=config.num_lobes).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.model.parameters(), lr=config.learning_rate)

        logger.info(f"VMF Guiding Distribution initialized with {config.num_lobes} lobes")

    @property
    def name(self) -> str:
        return "VMF Mixture"

    def _mixture(self, position, wi, normal, roughness) -> BatchedMixedSphericalGaussianDistribution:
        params = self.network(self._position_to_unit_cube(position),
                              to_unit(wi), to_unit(normal), roughness)
        return BatchedMixedSphericalGaussianDistribution(params)

    def sample(self, position, wi, normal, roughness) -> Tuple[torch.Tensor, torch.Tensor]:
        self.network.eval()
        with torch.no_grad():
            dist = self._mixture(position, wi, normal, roughness)
            directions = dist.sample()
            return directions, dist.pdf(directions)

    def pdf(self, position, wi, normal, roughness, wo) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            return self._mixture(position, wi, normal, roughness).pdf(wo)

    def train_step(self, batch: TrainingBatch) -> float:
        if not batch.is_valid():
            return -1.0

        weight = batch.guide_weight_rgb.mean(dim=1)  # F / p_sample
        valid = torch.isfinite(weight)
        sel = lambda t: t[valid]

        self.network.train()
        dist = self._mixture(sel(batch.position), sel(batch.wi), sel(batch.normal), sel(batch.roughness))
        loss = -(sel(weight).detach() * torch.log(dist.pdf(sel(batch.wo)))).mean()

        if not torch.isfinite(loss):
            logger.warning("Non-finite VMF loss; skipping step")
            return -1.0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class VMFIntegrator(PathGuidingIntegrator):
    distribution_class = VMFGuidingDistribution
    config_class = VMFConfig
    TRAINING_BUDGET_PERCENT = 25          # NPM trains for the first 25% of the budget
