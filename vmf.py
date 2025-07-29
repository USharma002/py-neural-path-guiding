import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from math_utils import *

M_EPSILON = 1e-5
M_INV_4PI = 0.07957747154594766788
M_2PI = 6.28318530717958647693

class VMFDistribution(nn.Module):
    def __init__(self, kappa):
        self.kappa = torch.as_tensor(kappa)

    def eval(self, cosTheta):
        cosTheta = torch.as_tensor(cosTheta)

        # Handle both scalar and tensor cosTheta
        if self.kappa.item() < M_EPSILON:
            return torch.full_like(cosTheta, M_INV_4PI)

        exponent = self.kappa * torch.minimum(torch.tensor(0.0, device=cosTheta.device), cosTheta - 1)
        norm = self.kappa / (M_2PI * (1 - torch.exp(-2 * self.kappa)))
        return torch.exp(exponent) * norm


    def eval_direction(self, wi):
        cosTheta = wi[..., 2]  # Works for shape [..., 3]
        return self.eval(cosTheta)

    def eval_cartesian(self, wi, mu):
        cosTheta = torch.sum(wi * mu, dim=-1)  # Dot product across batch
        return self.eval(cosTheta)

    def eval_spherical(self, wi, theta, phi):
        cartesian = spherical_to_cartesian(theta, phi)
        return self.eval_cartesian(wi, cartesian)

    def sample(self, u):
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


    def sample_cartesian(self, u, mu):
        local_dir = self.sample(u)
        return to_world(mu, local_dir) 


    def sample_spherical(self, u, theta, phi):
        local_dir = self.sample(u)
        mu_cartesian = spherical_to_cartesian(theta, phi)
        return to_world(mu_cartesian, local_dir)