"""Direction and position helpers."""
from __future__ import annotations

import math
from typing import Tuple

import torch

M_EPSILON = 1e-5


def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Polar angle theta in [0, pi] and azimuth phi in [0, 2pi] -> unit vector (..., 3)."""
    sin_theta = torch.sin(theta)
    return torch.stack((sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)), dim=-1)


def scene_bbox_tensors(scene, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scene bbox as (min, max) tensors; ScalarPoint3f has no DLPack, so components are copied."""
    bbox = scene.bbox()
    return (torch.tensor([bbox.min[i] for i in range(3)], dtype=torch.float32, device=device),
            torch.tensor([bbox.max[i] for i in range(3)], dtype=torch.float32, device=device))


def normalize_position(position: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    """World-space positions (N, 3) -> [0, 1]^3 within the scene bbox."""
    return ((position - bbox_min) / (bbox_max - bbox_min).clamp_min(1e-6)).clamp(0.0, 1.0)


def dir_to_square(d: torch.Tensor) -> torch.Tensor:
    """Equal-area (cylindrical) map of unit directions (N, 3) to [0,1]^2: u = (phi + pi) / 2pi, v = (1 - z) / 2."""
    phi = torch.atan2(d[:, 1], d[:, 0])
    return torch.stack([(phi + math.pi) / (2 * math.pi), (1.0 - d[:, 2]) / 2.0], dim=1).clamp(0.0, 1.0)


def square_to_dir(uv: torch.Tensor) -> torch.Tensor:
    """Inverse of dir_to_square; the map has constant Jacobian d_omega = 4*pi du dv."""
    phi = uv[:, 0] * (2 * math.pi) - math.pi
    z = 1.0 - 2.0 * uv[:, 1]
    sin_theta = torch.sqrt((1.0 - z * z).clamp(min=0.0))
    return torch.stack([sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), z], dim=1)


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """Map unit vectors from [-1,1] to [0,1], the input range tiny-cuda-nn encodings expect."""
    return (torch.nn.functional.normalize(x, dim=-1) + 1.0) * 0.5
