import torch
import torch.nn as nn
import math


from math_utils import (
    spherical_to_cartesian, cartesian_to_spherical, safe_sqrt, to_world
)

from guiding_network import GuidingNetwork
from vmf import *

M_EPSILON = 1e-5
M_INV_4PI = 1.0 / (4 * math.pi)
M_INV_2PI = 1.0 / (2 * math.pi)
M_2PI = 2 * math.pi
N_DIM_VMF = 4  # [lambda, kappa, theta, phi]

def vmf_sample_spherical(kappas, thetas, phis):
    device = kappas.device
    batch_size = kappas.shape[0]

    mu = spherical_to_cartesian(thetas, phis) # (B, 3)

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

class VMFKernel(nn.Module):
    def __init__(self, lambda_=0.0, kappa=0.0, theta=0.0, phi=0.0):
        super().__init__()
        self.lambda_ = torch.as_tensor(lambda_, dtype=torch.float32)
        self.kappa = torch.as_tensor(kappa, dtype=torch.float32)
        self.theta = torch.as_tensor(theta, dtype=torch.float32)
        self.phi = torch.as_tensor(phi, dtype=torch.float32)

    def get_spherical_dir(self):
        return torch.stack([self.theta, self.phi])

    def get_cartesian_dir(self):
        return spherical_to_cartesian(self.theta, self.phi)

    def sample(self, u):
        return VMFDistribution(self.kappa).sample_spherical(u, self.theta, self.phi)

    def eval(self, wi):
        return self.lambda_ * self.pdf(wi)

    def pdf(self, wi):
        return VMFDistribution(self.kappa).eval_spherical(wi, self.theta, self.phi)

    def product(self, other):
        def norm(kappa):
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


class MixedSphericalGaussianDistribution(nn.Module):
    def __init__(self, network_params, batch_idx=0):
        super().__init__()
        lambdas = network_params['lambda'][batch_idx].to(torch.float32)
        kappas = network_params['kappa'][batch_idx].to(torch.float32)
        thetas = network_params['theta'][batch_idx].to(torch.float32)
        phis = network_params['phi'][batch_idx].to(torch.float32)
        
        self.N = kappas.shape[0]
        self.kernels = nn.ModuleList()
        
        for i in range(self.N):
            kernel = VMFKernel(
                lambda_=lambdas[i],
                kappa=kappas[i],
                theta=thetas[i],
                phi=phis[i]
            )
            self.kernels.append(kernel)
            
        self.totalWeight = torch.sum(lambdas)
        if self.totalWeight < M_EPSILON:
            self.totalWeight = torch.tensor(M_EPSILON)
            
        self.weights = lambdas / self.totalWeight


    def eval(self, wi):
        wi_tensor = torch.as_tensor(wi, dtype=torch.float32, device=self.weights.device)
        return sum(kernel.eval(wi_tensor) for kernel in self.kernels)

    def pdf(self, wi):
        """Computes the normalized probability density function: sum(weight_i * pdf_i)."""
        wi_tensor = torch.as_tensor(wi, dtype=torch.float32, device=self.weights.device)
        return sum(self.weights[i] * self.kernels[i].pdf(wi_tensor) for i in range(self.N))

    def apply_cosine_lobe(self, normal):
        normal_tensor = torch.as_tensor(normal, dtype=torch.float32, device=self.weights.device)
        sph_normal = cartesian_to_spherical(normal_tensor)
        cosine_vmf = VMFKernel(1.0, self.VMF_DIFFUSE_LOBE, sph_normal[0], sph_normal[1])

        new_lambdas = []
        for kernel in self.kernels:
            kernel.product(cosine_vmf)
            new_lambdas.append(kernel.lambda_)
        
        self.totalWeight = torch.sum(torch.stack(new_lambdas))
        if self.totalWeight < M_EPSILON:
            self.totalWeight = torch.tensor(M_EPSILON)
        self.weights = torch.stack(new_lambdas) / self.totalWeight


class BatchedMixedSphericalGaussianDistribution(nn.Module):
    def __init__(self, network_params):
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


    def _get_lobe_pdfs(self, wi):
        # --- No changes needed here ---
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

    def pdf(self, wi):
        """
        Computes the normalized PDF for an entire batch of directions.
        """
        # Get the PDFs for all lobes. Shape: (B, K)
        lobe_pdfs = self._get_lobe_pdfs(wi)
        
        # Multiply by normalized mixture weights and sum along the K dimension.
        # Final shape: (B,)
        return torch.sum(self.weights * lobe_pdfs, dim=1)

    def eval(self, wi):
        """
        Computes the unnormalized, scaled value for an entire batch of directions.
        """
        # Get the PDFs for all lobes. Shape: (B, K)
        lobe_pdfs = self._get_lobe_pdfs(wi)
        
        # Multiply by unnormalized lambdas and sum along the K dimension.
        # Final shape: (B,)
        return torch.sum(self.lambdas * lobe_pdfs, dim=1)

    def sample(self):
        lobe_indices = torch.multinomial(self.weights, num_samples=1)

        chosen_kappas = torch.gather(self.kappas, 1, lobe_indices).squeeze(1)
        chosen_thetas = torch.gather(self.thetas, 1, lobe_indices).squeeze(1)
        chosen_phis = torch.gather(self.phis, 1, lobe_indices).squeeze(1)
        
        # Sample from the selected vMF distributions.
        return vmf_sample_spherical(chosen_kappas, chosen_thetas, chosen_phis)

    def entropy(self):

        epsilon = 1e-8
        H = -torch.sum(self.weights * torch.log(self.weights + epsilon), dim=-1)
        return H