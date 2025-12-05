"""Von Mises-Fisher mixture distributions for path guiding."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from guiding.config import get_logger
from utils.math_utils import spherical_to_cartesian, cartesian_to_spherical
from distributions.vmf import VMFDistribution

logger = get_logger("vmf_mixture")

# Constants
M_EPSILON = 1e-5
M_INV_4PI = 1.0 / (4.0 * math.pi)
M_INV_2PI = 1.0 / (2.0 * math.pi)
M_2PI = 2.0 * math.pi
N_DIM_VMF = 4  # [lambda, kappa, theta, phi]

# Diffuse lobe approximation for cosine-weighted hemisphere sampling
VMF_DIFFUSE_LOBE_KAPPA = 2.188


def vmf_sample_spherical(
    kappas: torch.Tensor, 
    thetas: torch.Tensor, 
    phis: torch.Tensor
) -> torch.Tensor:
    """Sample directions from vMF distributions with given parameters.
    
    Args:
        kappas: Concentration parameters (B,)
        thetas: Polar angles of mean directions (B,)
        phis: Azimuthal angles of mean directions (B,)
        
    Returns:
        Sampled directions (B, 3)
    """
    device = kappas.device
    batch_size = kappas.shape[0]

    mu = spherical_to_cartesian(thetas, phis)  # (B, 3)

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

# Diffuse lobe approximation for cosine-weighted hemisphere sampling
VMF_DIFFUSE_LOBE_KAPPA = 2.188


class VMFKernel(nn.Module):
    """A single von Mises-Fisher kernel with weight, concentration, and direction."""
    
    def __init__(self, lambda_: float = 0.0, kappa: float = 0.0, theta: float = 0.0, phi: float = 0.0):
        super().__init__()
        self.lambda_ = torch.as_tensor(lambda_, dtype=torch.float32)
        self.kappa = torch.as_tensor(kappa, dtype=torch.float32)
        self.theta = torch.as_tensor(theta, dtype=torch.float32)
        self.phi = torch.as_tensor(phi, dtype=torch.float32)

    def get_spherical_dir(self) -> torch.Tensor:
        """Get the mean direction in spherical coordinates."""
        return torch.stack([self.theta, self.phi])

    def get_cartesian_dir(self) -> torch.Tensor:
        """Get the mean direction in Cartesian coordinates."""
        return spherical_to_cartesian(self.theta, self.phi)

    def sample(self, u: torch.Tensor) -> torch.Tensor:
        """Sample a direction from this kernel."""
        return VMFDistribution(self.kappa).sample_spherical(u, self.theta, self.phi)

    def eval(self, wi: torch.Tensor) -> torch.Tensor:
        """Evaluate the weighted PDF."""
        return self.lambda_ * self.pdf(wi)

    def pdf(self, wi: torch.Tensor) -> torch.Tensor:
        """Evaluate the normalized PDF."""
        return VMFDistribution(self.kappa).eval_spherical(wi, self.theta, self.phi)

    def product(self, other: VMFKernel) -> None:
        """Compute the product of this kernel with another (in-place)."""
        def norm(kappa: torch.Tensor) -> torch.Tensor:
            kappa = torch.clamp(kappa, min=1e-8)
            return torch.where(
                kappa < self.min_valid_kappa,
                torch.tensor(M_INV_4PI),
                kappa * M_INV_2PI / (1 - torch.exp(-2 * kappa))
            )

        mu_a = self.get_cartesian_dir()
        mu_b = other.get_cartesian_dir()

        new_mu = mu_a * self.kappa + mu_b * other.kappa
        new_kappa = torch.linalg.norm(new_mu)

        if new_kappa >= 1e-3:
            new_mu = new_mu / new_kappa
        else:
            new_kappa = torch.tensor(0.0)
            new_mu = mu_a

        theta_new, phi_new = cartesian_to_spherical(new_mu)

        dot_a = torch.sum(new_mu * mu_a)
        dot_b = torch.sum(new_mu * mu_b)

        e = torch.exp(self.kappa * (dot_a - 1) + other.kappa * (dot_b - 1))
        s = norm(self.kappa) * norm(other.kappa) / norm(new_kappa) * e

        self.lambda_ = s * self.lambda_ * other.lambda_
        self.kappa = new_kappa
        self.theta = theta_new
        self.phi = phi_new

    min_valid_kappa = 1e-5


class BatchedMixedSphericalGaussianDistribution(nn.Module):
    """Batched mixture of vMF distributions for efficient GPU computation.
    
    Args:
        network_params: Dictionary with 'lambda', 'kappa', 'theta', 'phi' tensors
                       of shape (batch_size, K)
    """
    
    def __init__(self, network_params: Dict[str, torch.Tensor]):
        super().__init__()
    
        # (batch_size, K)
        lambdas_non_negative = F.relu(network_params['lambda'].to(torch.float32))
        self.lambdas = lambdas_non_negative + 1e-6
        self.kappas = network_params['kappa'].to(torch.float32)
        self.thetas = network_params['theta'].to(torch.float32)
        self.phis = network_params['phi'].to(torch.float32)
        
        self.batch_size, self.K = self.lambdas.shape
        
        self.totalWeight = torch.sum(self.lambdas, dim=1)
        self.totalWeight = torch.clamp(self.totalWeight, min=M_EPSILON)
        
        self.weights = self.lambdas / self.totalWeight.unsqueeze(1)

    def _get_lobe_pdfs(self, wi: torch.Tensor) -> torch.Tensor:
        """Compute PDFs for all lobes in the mixture.
        
        Args:
            wi: Direction vectors of shape (B, 3)
            
        Returns:
            Lobe PDFs of shape (B, K)
        """
        B, _ = wi.shape
        mus = spherical_to_cartesian(self.thetas, self.phis)
        dot_products = torch.sum(mus * wi.unsqueeze(1), dim=2)

        # ======================================================================
        #  ROBUST NORMALIZATION CALCULATION
        # ======================================================================

        # 1. Identify where kappa is too small to be stable.
        #    Use a slightly larger threshold for safety.
        is_small_kappa = self.kappas < 1e-5
        
        # 2. For the stable kappas, calculate the normalization constant.
        #    We use a "safe" version of kappa for the calculation to avoid issues,
        #    but the original kappa for the mask.
        safe_kappas = torch.clamp(self.kappas, min=1e-6)
        denominator = 2 * math.pi * (1 - torch.exp(-2 * safe_kappas))
        
        # Avoid division by zero by setting the denominator to 1 where kappa is small.
        # The result of this division will be ignored for these elements anyway.
        denominator[is_small_kappa] = 1.0
        
        # Now this division is always safe.
        C_kappa = safe_kappas / denominator

        # 3. For the small kappas, the PDF should be that of a uniform distribution.
        #    Use torch.where to select the correct normalization constant.
        #    Since the division was made safe, this will no longer receive a NaN.
        uniform_pdf_const = 1.0 / (4 * math.pi)
        C_kappa = torch.where(is_small_kappa, uniform_pdf_const, C_kappa)

        # 4. Calculate the exponential term. This part is already stable.
        #    exp(k * (dot - 1)) = exp(k*dot) * exp(-k)
        exp_term = torch.exp(self.kappas * (dot_products - 1))
        
        # 5. Final PDF. For the uniform case, where kappa is near zero, exp_term is ~1,
        #    so the PDF correctly becomes ~1 / (4*pi).
        lobe_pdfs = C_kappa * exp_term
        
        return lobe_pdfs

    def pdf(self, wi: torch.Tensor) -> torch.Tensor:
        """Compute the normalized PDF for a batch of directions.
        
        Args:
            wi: Direction vectors of shape (B, 3)
            
        Returns:
            PDF values of shape (B,)
        """
        lobe_pdfs = self._get_lobe_pdfs(wi)
        return torch.sum(self.weights * lobe_pdfs, dim=1)

    def eval(self, wi: torch.Tensor) -> torch.Tensor:
        """Compute the unnormalized weighted PDF for a batch of directions.
        
        Args:
            wi: Direction vectors of shape (B, 3)
            
        Returns:
            Weighted PDF values of shape (B,)
        """
        lobe_pdfs = self._get_lobe_pdfs(wi)
        return torch.sum(self.lambdas * lobe_pdfs, dim=1)

    def sample(self) -> torch.Tensor:
        """Sample directions from the mixture distribution.
        
        Returns:
            Sampled directions of shape (B, 3)
        """

        # Ensure weights are a float tensor on same device
        weights = self.weights
        device = weights.device
        weights = weights.to(torch.float32)

        # If 1D, interpret as batch size 1
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)  # (1, K)

        B, K = weights.shape

        # Diagnostic checks
        n_nan = torch.isnan(weights).sum().item()
        n_inf = torch.isinf(weights).sum().item()
        n_neg = (weights < 0).sum().item()
        row_sums = weights.sum(dim=1)
        n_zero_rows = (row_sums <= 1e-12).sum().item()

        # if n_nan or n_inf or n_neg or n_zero_rows:
        #     print("=== BatchedMixedSphericalGaussianDistribution.sample() DIAGNOSTIC ===")
        #     print(f"weights shape: {weights.shape} device: {weights.device}")
        #     print(f"n_nan: {n_nan}, n_inf: {n_inf}, n_neg: {n_neg}, n_zero_rows: {n_zero_rows}")
        #     print(f"row_sums.min: {float(row_sums.min()) if row_sums.numel() else 'N/A'}, row_sums.max: {float(row_sums.max()) if row_sums.numel() else 'N/A'}")
        #     # print the first few rows (move to CPU for safety)
        #     print("weights[0:6]:\n", weights[:6].detach().cpu().numpy())
        #     print("=====================================================================")

        # Fix NaN/Inf: replace with zeros (we'll restore uniform if entire row zero)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))

        # Clamp negatives to zero
        weights = torch.clamp(weights, min=0.0)

        # If entire row sums to zero, set uniform small positive weights
        row_sums = weights.sum(dim=1, keepdim=True)  # (B,1)
        zero_mask = (row_sums <= 1e-12).squeeze(1)    # (B,)
        if zero_mask.any():
            # create uniform weights for those rows
            num_zero = int(zero_mask.sum().item())
            uniform = (1.0 / float(K)) * torch.ones((num_zero, K), device=device, dtype=weights.dtype)
            weights[zero_mask] = uniform
            row_sums = weights.sum(dim=1, keepdim=True)

        # Normalize to probabilities (stable)
        probs = weights / (row_sums + 1e-12)

        # Try to sample with multinomial on device, fall back to CPU categorical on failure
        try:
            lobe_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
        except RuntimeError as e:
            logger.warning(f"torch.multinomial failed on device; falling back to CPU Categorical. Error: {e}")
            probs_cpu = probs.cpu()
            cat = Categorical(probs_cpu)
            lobe_indices = cat.sample().to(device)

        # Gather chosen vMF parameters per-batch
        # lobe_indices shape: (B,) -> need shape (B,1) to gather along dim=1
        idx = lobe_indices.unsqueeze(1)
        chosen_kappas = torch.gather(self.kappas, 1, idx).squeeze(1)   # (B,)
        chosen_thetas = torch.gather(self.thetas, 1, idx).squeeze(1)   # (B,)
        chosen_phis   = torch.gather(self.phis, 1, idx).squeeze(1)     # (B,)

        # Sample directions from selected vmf lobes
        sampled = vmf_sample_spherical(chosen_kappas, chosen_thetas, chosen_phis)  # (B, 3)

        return sampled

    def entropy(self) -> torch.Tensor:
        """Compute the entropy of the mixture weights.
        
        Returns:
            Entropy value for each sample in the batch
        """
        epsilon = M_EPSILON
        H = -torch.sum(self.weights * torch.log(self.weights + epsilon), dim=-1)
        return H