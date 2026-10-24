"""Training batch shared by all guiding methods and the NRC."""

from __future__ import annotations

from dataclasses import dataclass, fields
import torch


@dataclass
class TrainingBatch:
    """One row per recorded path vertex; methods needing more fields subclass this."""
    position: torch.Tensor          # (N, 3) world
    normal: torch.Tensor            # (N, 3) world
    wi: torch.Tensor                # (N, 3) local
    wo: torch.Tensor                # (N, 3) local
    roughness: torch.Tensor         # (N, 1)
    combined_pdf: torch.Tensor      # (N,) MIS mixture pdf that generated wo
    guide_weight_rgb: torch.Tensor  # (N, 3) f * cos * L_i(wo) / combined_pdf: MC weight of the product target
    outgoing_rgb: torch.Tensor      # (N, 3) outgoing radiance toward wi (NRC target)
    num_samples: int

    @classmethod
    def empty(cls, device: str = "cuda") -> "TrainingBatch":
        tensors = {f.name: torch.empty(0, device=device) for f in fields(cls) if f.name != "num_samples"}
        return cls(**tensors, num_samples=0)

    def subset(self, index: torch.Tensor) -> "TrainingBatch":
        """Rows at the given integer indices."""
        tensors = {f.name: getattr(self, f.name)[index] for f in fields(self) if f.name != "num_samples"}
        return type(self)(**tensors, num_samples=int(index.numel()))

    def is_valid(self) -> bool:
        return self.num_samples > 0
