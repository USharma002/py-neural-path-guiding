"""Von Mises-Fisher distribution implementation for directional statistics."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.math_utils import M_EPSILON, M_2PI, M_INV_4PI, safe_sqrt, spherical_to_cartesian, to_world, uniform_sample_sphere


class VMFDistribution(nn.Module):
    """Von Mises-Fisher distribution on the unit sphere.
    
    The vMF distribution is the spherical analogue of the normal distribution,
    parameterized by a mean direction and concentration parameter kappa.
    
    Args:
        kappa: Concentration parameter (higher = more concentrated)
    """
    
    def __init__(self, kappa: float) -> None:
        self.kappa = torch.as_tensor(kappa)

    def eval(self, cosTheta: torch.Tensor) -> torch.Tensor:
        """Evaluate the PDF given the cosine of the angle from the mean direction.
        
        Args:
            cosTheta: Cosine of angle from mean direction
            
        Returns:
            PDF value(s)
        """
        cosTheta = torch.as_tensor(cosTheta)

        # Handle both scalar and tensor cosTheta
        if self.kappa.item() < M_EPSILON:
            return torch.full_like(cosTheta, M_INV_4PI)

        exponent = self.kappa * torch.minimum(torch.tensor(0.0, device=cosTheta.device), cosTheta - 1)
        norm = self.kappa / (M_2PI * (1 - torch.exp(-2 * self.kappa)))
        return torch.exp(exponent) * norm

    def eval_direction(self, wi: torch.Tensor) -> torch.Tensor:
        """Evaluate the PDF for a direction vector (assumes z-aligned mean)."""
        cosTheta = wi[..., 2]
        return self.eval(cosTheta)

    def eval_cartesian(self, wi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Evaluate the PDF for direction wi with mean direction mu."""
        cosTheta = torch.sum(wi * mu, dim=-1)
        return self.eval(cosTheta)

    def eval_spherical(self, wi: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Evaluate the PDF with mean direction given in spherical coordinates."""
        cartesian = spherical_to_cartesian(theta, phi)
        return self.eval_cartesian(wi, cartesian)

    def sample(self, u: torch.Tensor) -> torch.Tensor:
        """Sample a direction using uniform random numbers.
        
        Args:
            u: Uniform random values of shape (..., 2)
            
        Returns:
            Sampled direction vectors of shape (..., 3)
        """
        if self.kappa.item() < M_EPSILON:
            return uniform_sample_sphere(u)

        cosTheta = 1 + torch.log1p(-u[..., 0] + torch.exp(-2 * self.kappa) * u[..., 0]) / self.kappa
        sinTheta = safe_sqrt(1 - cosTheta ** 2)
        phi = M_2PI * u[..., 1]
        sinPhi = torch.sin(phi)
        cosPhi = torch.cos(phi)

        sample = torch.stack([
            cosPhi * sinTheta, 
            sinPhi * sinTheta, 
            cosTheta], 
        dim=-1)

        return F.normalize(sample, dim=-1)

    def sample_cartesian(self, u: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Sample with mean direction given as Cartesian vector."""
        local_dir = self.sample(u)
        return to_world(mu, local_dir)

    def sample_spherical(self, u: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Sample with mean direction given in spherical coordinates."""
        local_dir = self.sample(u)
        mu_cartesian = spherical_to_cartesian(theta, phi)
        return to_world(mu_cartesian, local_dir)