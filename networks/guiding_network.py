"""tiny-cuda-nn network predicting vMF mixture parameters (Neural Parametric Mixtures, Dong et al. 2023)."""
from __future__ import annotations

import math
from typing import Dict

import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidingNetwork(nn.Module):
    """(pos in [0,1]^3, wi, normal, roughness) -> (lambda, kappa, theta, phi) for K lobes.

    Position: 8-level dense grid, 4 features per level, resolutions 8..86; wi and normal:
    spherical harmonics; roughness: raw auxiliary feature; decoder: 3-layer MLP of width 64.
    """
    def __init__(self, device: str, K: int = 8):
        super().__init__()
        self.device = device
        self.K = K

        encoding = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 3, "otype": "DenseGrid", "n_levels": 8, "n_features_per_level": 4,
                 "base_resolution": 8, "per_level_scale": (86 / 8) ** (1 / 7)},
                {"n_dims_to_encode": 3, "otype": "SphericalHarmonics", "degree": 4},
                {"n_dims_to_encode": 3, "otype": "SphericalHarmonics", "degree": 4},
                {"n_dims_to_encode": 1, "otype": "Identity"},
            ],
        }
        network = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                   "n_neurons": 64, "n_hidden_layers": 3}

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=10, n_output_dims=4 * self.K, encoding_config=encoding, network_config=network,
        ).to(self.device)

    def network_output_to_params(self, network_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(N, 4K) raw output -> (N, K) tensors: lambda = softmax, kappa = exp, theta / phi via sigmoid."""
        raw = network_output.float().view(-1, self.K, 4)
        return {
            "lambda": F.softmax(raw[..., 0], dim=-1),
            "kappa": torch.exp(raw[..., 1]),
            "theta": torch.sigmoid(raw[..., 2]) * math.pi,
            "phi": torch.sigmoid(raw[..., 3]) * 2 * math.pi,
        }

    def forward(self, pos01, wi01, normal01, roughness) -> Dict[str, torch.Tensor]:
        """All inputs already mapped to [0, 1]."""
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)
        x = torch.cat([pos01, wi01, normal01, roughness], dim=1).contiguous()
        return self.network_output_to_params(self.model(x))
