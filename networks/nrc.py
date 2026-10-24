"""Neural Radiance Cache (Muller et al. 2021)."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import tinycudann as tcnn

from guiding.training_data import TrainingBatch
from utils.math_utils import normalize_position, scene_bbox_tensors, to_unit


class NeuralRadianceCache(nn.Module):
    """Predicts outgoing RGB radiance from (world position, local view direction, normal, roughness)."""

    def __init__(self, device: str = "cuda", learning_rate: float = 1e-2) -> None:
        super().__init__()
        self.device = device
        self.bbox_min: Optional[torch.Tensor] = None
        self.bbox_max: Optional[torch.Tensor] = None

        encoding = {"otype": "Composite", "nested": [
            {"n_dims_to_encode": 3, "otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2,
             "log2_hashmap_size": 19, "base_resolution": 16, "per_level_scale": 1.5},
            {"n_dims_to_encode": 3, "otype": "SphericalHarmonics", "degree": 4},
            {"n_dims_to_encode": 3, "otype": "SphericalHarmonics", "degree": 4},
            {"n_dims_to_encode": 1, "otype": "OneBlob", "n_bins": 4},
        ]}
        network = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                   "n_neurons": 64, "n_hidden_layers": 4}
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=10, n_output_dims=3, encoding_config=encoding, network_config=network,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-15)

    def set_scene(self, scene: Any) -> None:
        self.bbox_min, self.bbox_max = scene_bbox_tensors(scene, self.device)

    def forward(self, positions, view_dirs, normals, roughness) -> torch.Tensor:
        x = torch.cat([normalize_position(positions, self.bbox_min, self.bbox_max),
                       to_unit(view_dirs), to_unit(normals), roughness], dim=1)
        return torch.nn.functional.softplus(self.model(x.float().contiguous()).float())

    def query(self, positions, normals, view_dirs, roughness) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.forward(positions, view_dirs, normals, roughness)

    def train_step_from_batch(self, batch: TrainingBatch) -> float:
        if not batch.is_valid():
            return -1.0
        self.model.train()
        pred = self.forward(batch.position, batch.wi, batch.normal, batch.roughness)
        # Relative L2 loss: normalized by the squared prediction, without gradient
        loss = torch.mean((pred - batch.outgoing_rgb).square() / (pred.detach().square() + 1e-2))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
