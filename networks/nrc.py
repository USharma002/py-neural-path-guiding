"""Neural Radiance Cache (NRC)."""

from __future__ import annotations

from typing import Optional, Any

import torch
import torch.nn as nn
import tinycudann as tcnn

from guiding.config import get_logger
from guiding.training_data import TrainingBatch

logger = get_logger("nrc")


class NeuralRadianceCache(nn.Module):
    """
    NRC predicts RGB incoming radiance given:
      - position (world, normalized internally to [0,1]^3 via bbox)
      - view direction (IMPORTANT: this must be in the SAME frame as training;
        your integrator stores wi in local shading frame)
      - normal (world or local is OK as long as consistent; you train with si.n)
      - roughness

    Output is forced positive via softplus.
    """

    def __init__(self, device: str = "cuda", learning_rate: float = 1e-2) -> None:
        super().__init__()
        self.device = device

        # bbox for world->normalized position
        self.bbox_min: Optional[torch.Tensor] = None
        self.bbox_max: Optional[torch.Tensor] = None
        self._warned_missing_bbox: bool = False

        input_dims = 10  # pos(3) + view(3) + normal(3) + rough(1)
        output_dims = 3  # RGB

        # TinyCUDA-NN composite encoding + MLP
        config = {
            "encoding": {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "HashGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": 16,
                        "per_level_scale": 1.5,
                    },
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    {
                        "n_dims_to_encode": 1,
                        "otype": "OneBlob",
                        "n_bins": 4,
                    },
                ],
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 4,
            },
        }

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=input_dims,
            n_output_dims=output_dims,
            encoding_config=config["encoding"],
            network_config=config["network"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-15)

        logger.info("Neural Radiance Cache initialized.")

    # -------------------- Scene / bbox --------------------

    def set_scene(self, scene: Any) -> None:
        """Store bbox tensors from a Mitsuba scene."""
        if scene is None:
            return
        bbox = scene.bbox()
        bbox_min_list = [bbox.min[i] for i in range(3)]
        bbox_max_list = [bbox.max[i] for i in range(3)]
        self.bbox_min = torch.tensor(bbox_min_list, dtype=torch.float32, device=self.device)
        self.bbox_max = torch.tensor(bbox_max_list, dtype=torch.float32, device=self.device)

    def set_bbox(self, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> None:
        """Directly set bbox tensors (already torch)."""
        self.bbox_min = bbox_min.to(self.device).float()
        self.bbox_max = bbox_max.to(self.device).float()

    def _normalize_position(
        self,
        pos_world: torch.Tensor,
        bbox_min: Optional[torch.Tensor] = None,
        bbox_max: Optional[torch.Tensor] = None,
        *,
        require_bbox: bool,
    ) -> torch.Tensor:
        """World -> [0,1]^3 for HashGrid."""
        if bbox_min is None:
            bbox_min = self.bbox_min
        if bbox_max is None:
            bbox_max = self.bbox_max

        if bbox_min is None or bbox_max is None:
            if require_bbox:
                raise RuntimeError("NRC bbox not set (call set_scene/set_bbox before training).")
            # Query-time fallback: assume already normalized-ish
            if not self._warned_missing_bbox:
                logger.warning("NRC bbox missing; clamping positions to [0,1].")
                self._warned_missing_bbox = True
            return torch.clamp(pos_world, 0.0, 1.0)

        extent = (bbox_max - bbox_min).clamp_min(1e-6)
        pos = (pos_world - bbox_min) / extent
        return torch.clamp(pos, 0.0, 1.0)

    # -------------------- Input packing --------------------

    @staticmethod
    def _unit(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1)

    def _prepare_input(
        self,
        pos_01: torch.Tensor,
        view_dirs: torch.Tensor,
        normals: torch.Tensor,
        roughness: torch.Tensor,
    ) -> torch.Tensor:
        """(N,3)+(N,3)+(N,3)+(N,1) -> (N,10) contiguous float32."""
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)

        view_dirs = self._unit(view_dirs)
        normals = self._unit(normals)

        x = torch.cat([pos_01, view_dirs, normals, roughness], dim=1)
        return x.to(dtype=torch.float32, device=self.device).contiguous()

    # -------------------- Forward / Query --------------------

    def forward(
        self,
        pos_01: torch.Tensor,
        view_dirs: torch.Tensor,
        normals: torch.Tensor,
        roughness: torch.Tensor,
    ) -> torch.Tensor:
        x = self._prepare_input(pos_01, view_dirs, normals, roughness)
        y = self.model(x)
        return torch.nn.functional.softplus(y.float())

    def query(
        self,
        positions: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        roughness: torch.Tensor,
        *,
        bbox_min: Optional[torch.Tensor] = None,
        bbox_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference (positions are WORLD space here; normalized internally).
        NOTE: view_dirs must be in the same frame used during training.
        """
        self.model.eval()
        with torch.no_grad():
            pos_01 = self._normalize_position(
                positions.to(self.device),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                require_bbox=False,
            )
            x = self._prepare_input(pos_01, view_dirs, normals, roughness)
            y = self.model(x)
            return torch.nn.functional.softplus(y.float())

    # -------------------- Training --------------------

    def train_step_from_batch(self, batch: TrainingBatch) -> float:
        """
        Train step using your shared TrainingBatch.
        batch.position is WORLD space, so bbox must be set.
        """
        if not batch.is_valid():
            return -1.0

        self.model.train()

        pos_01 = self._normalize_position(
            batch.position.to(self.device),
            require_bbox=True,
        )

        pred = self.forward(
            pos_01,
            batch.wi.to(self.device),
            batch.normal.to(self.device),
            batch.roughness.to(self.device),
        )

        targets = batch.radiance_rgb.to(self.device)

        # Stable relative L2: weight by target magnitude (prevents "blow up to white")
        denom = targets.detach().square() + 1e-2
        loss = torch.mean((pred - targets).square() / denom)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
