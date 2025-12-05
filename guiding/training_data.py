"""Shared training data structures for neural path guiding systems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainingBatch:
    """A batch of training data shared between NRC and guiding systems.
    
    This dataclass holds all the information needed for training both
    the Neural Radiance Cache and the path guiding distribution.
    
    All tensors have shape (N, ...) where N is the number of valid samples.
    """
    # Core geometry (all in world space)
    position: torch.Tensor          # (N, 3) - surface position
    normal: torch.Tensor            # (N, 3) - surface normal
    wi: torch.Tensor                # (N, 3) - incoming direction (from camera)
    wo: torch.Tensor                # (N, 3) - outgoing direction (sampled)
    
    # For guiding distribution training
    position_normalized: torch.Tensor  # (N, 3) - position in [0,1]^3 or [-1,1]^3
    roughness: torch.Tensor            # (N, 1) - surface roughness
    combined_pdf: torch.Tensor         # (N,) - MIS combined PDF
    
    # Radiance targets
    radiance_scalar: torch.Tensor   # (N,) - scalar radiance (luminance)
    radiance_rgb: torch.Tensor      # (N, 3) - RGB radiance (product field)
    
    # Metadata
    num_samples: int
    device: str = "cuda"
    
    @classmethod
    def empty(cls, device: str = "cuda") -> "TrainingBatch":
        """Create an empty training batch."""
        return cls(
            position=torch.empty(0, 3, device=device),
            normal=torch.empty(0, 3, device=device),
            wi=torch.empty(0, 3, device=device),
            wo=torch.empty(0, 3, device=device),
            position_normalized=torch.empty(0, 3, device=device),
            roughness=torch.empty(0, 1, device=device),
            combined_pdf=torch.empty(0, device=device),
            radiance_scalar=torch.empty(0, device=device),
            radiance_rgb=torch.empty(0, 3, device=device),
            num_samples=0,
            device=device
        )
    
    def is_valid(self) -> bool:
        """Check if batch contains valid training data."""
        return self.num_samples > 0


def prepare_shared_training_data(
    integrator,  # PathGuidingIntegrator
    device: str = "cuda"
) -> TrainingBatch:
    """Prepare training data that can be shared between NRC and guiding systems.
    
    This function:
    1. Calls scatter_data_into_buffer() once (idempotent)
    2. Extracts and validates all tensors
    3. Returns a TrainingBatch usable by both systems
    
    Args:
        integrator: The path guiding integrator with surface interaction data
        device: Device for tensors
        
    Returns:
        TrainingBatch with all validated training data
    """
    import drjit as dr
    
    rec = integrator.surfaceInteractionRecord
    
    # Scatter once (idempotent - skips if already done)
    integrator.scatter_data_into_buffer()
    
    # Check for empty data
    if dr.width(rec.position) == 0:
        return TrainingBatch.empty(device)
    
    # Extract tensors
    pos = rec.position.torch().to(device)
    normal = rec.normal.torch().to(device)
    wi = rec.wi.torch().to(device)
    wo = rec.wo.torch().to(device)
    combined_pdf = rec.woPdf.torch().to(device)
    
    # RGB radiance from product field (bsdf * Li)
    radiance_rgb = rec.product.torch().to(device)
    
    # Scalar radiance (luminance)
    radiance_scalar = rec.radiance.torch().to(device)
    if radiance_scalar.ndim > 1:
        radiance_scalar = radiance_scalar.squeeze(-1)
    
    # Build validity mask
    valid_mask = torch.ones(pos.shape[0], dtype=torch.bool, device=device)
    
    # Filter NaN/Inf positions
    valid_mask &= ~torch.any(torch.isinf(pos) | torch.isnan(pos), dim=1)
    valid_mask &= ~torch.any(torch.isinf(normal) | torch.isnan(normal), dim=1)
    valid_mask &= ~torch.any(torch.isinf(wi) | torch.isnan(wi), dim=1)
    valid_mask &= ~torch.any(torch.isinf(wo) | torch.isnan(wo), dim=1)
    valid_mask &= ~torch.any(torch.isinf(radiance_rgb) | torch.isnan(radiance_rgb), dim=1)
    
    # Filter zero/invalid directions
    wi_len_sq = torch.sum(wi * wi, dim=1)
    valid_mask &= (wi_len_sq > 0.5) & (wi_len_sq < 1.5)
    
    wo_len_sq = torch.sum(wo * wo, dim=1)
    valid_mask &= (wo_len_sq > 0.5) & (wo_len_sq < 1.5)
    
    # Filter extreme radiance
    radiance_magnitude = torch.sum(radiance_rgb, dim=1)
    valid_mask &= (radiance_magnitude >= 0) & (radiance_magnitude < 1000)
    
    # Filter invalid PDFs
    valid_mask &= (combined_pdf > 0) & ~torch.isnan(combined_pdf) & ~torch.isinf(combined_pdf)
    
    if not torch.any(valid_mask):
        return TrainingBatch.empty(device)
    
    # Apply mask
    pos = pos[valid_mask]
    normal = normal[valid_mask]
    wi = wi[valid_mask]
    wo = wo[valid_mask]
    radiance_rgb = radiance_rgb[valid_mask]
    radiance_scalar = radiance_scalar[valid_mask] if radiance_scalar.numel() > 0 else torch.zeros(pos.shape[0], device=device)
    combined_pdf = combined_pdf[valid_mask]
    
    # Normalize position to [0, 1]^3 using scene bbox
    bbox_min = integrator.bbox_min.torch().to(device)
    bbox_max = integrator.bbox_max.torch().to(device)
    pos_normalized = (pos - bbox_min) / (bbox_max - bbox_min + 1e-6)
    
    # Create roughness (placeholder - could extract from BSDF)
    roughness = torch.ones((pos.shape[0], 1), device=device)
    
    return TrainingBatch(
        position=pos,
        normal=normal,
        wi=wi,
        wo=wo,
        position_normalized=pos_normalized,
        roughness=roughness,
        combined_pdf=combined_pdf,
        radiance_scalar=radiance_scalar,
        radiance_rgb=radiance_rgb,
        num_samples=pos.shape[0],
        device=device
    )
