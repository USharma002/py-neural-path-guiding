import torch
import torch.nn as nn
import tinycudann as tcnn
import math
import torch.nn.functional as F

import drjit as dr

class NeuralRadianceCache(nn.Module):
    def __init__(self, device, learning_rate: float = 2e-3):
        super().__init__()
        self.device = device

        # Total input dimensions = 3 (pos) + 3 (normal) + 3 (dir) + 1 (roughness) = 7
        input_dims = 10
        # We need 4 raw parameters per lobe: (lambda, kappa, theta, phi)
        output_dims = 3
        
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
                        "dtype": "float32",  
                    },
                    {
                        "n_dims_to_encode": 4,
                        "otype": "Identity",
                        "dtype": "float32",  
                    }
                ]
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU", # IMPORTANT: Activations are applied manually
                "n_neurons": 128,
                "n_hidden_layers": 8,
            }
        }

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=input_dims,
            n_output_dims=output_dims,
            encoding_config=config["encoding"],
            network_config=config["network"],
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, eps=1e-15)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        print("Neural Radiance Cache initialized.")

    def _prepare_input(self, positions, normals, view_dirs, roughness):
        """Helper to concatenate and validate inputs."""
        # A more robust implementation would normalize position based on the scene AABB
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)

        positions = (positions + 1.0) / 2.0

        input_tensor = torch.cat([positions, normals, view_dirs, roughness], dim=1)
        return input_tensor

    def query(self, positions, normals, view_dirs, roughness):
        """
        Queries the network and returns processed vMF mixture parameters.
        """
        # Set to evaluation mode and disable gradients for inference speed
        self.model.eval()
        with torch.no_grad():
            input_tensor = self._prepare_input(positions, normals, view_dirs, roughness)
            # The network expects inputs in the [0, 1] range.
            # Normalizing with the scene AABB is the proper way to do this.
            # A simple (pos + 1)/2 is a placeholder for a [-1, 1] world.
            raw_output = self.model(input_tensor.to(self.device))
        
        # Process the raw output to get meaningful parameters
        return raw_output.float()

    def forward(self, positions, normals, view_dirs, roughness):
        """
        A forward pass for TRAINING. Gradients are tracked.
        """
        # Note: No model.eval() and no torch.no_grad()
        input_tensor = self._prepare_input(positions, normals, view_dirs, roughness)

        # The tcnn model handles the device placement
        raw_output = self.model(input_tensor.contiguous())
        
        return raw_output.float()

    def prepare_training_data(self, integrator):
        """
        Processes the integrator's record to produce clean, validated training tensors.
        """
        rec = integrator.surfaceInteractionRecord
        # integrator.scatter_data_into_buffer()

        if dr.width(rec.position) == 0:
            return (None,) * 6

        pos = rec.position.torch()
        wi = rec.wi.torch() # Direction of incoming ray
        wo = rec.wo_world.torch() # direction sampled using bsdf

        targets_li = rec.radiance.torch()
        normal = rec.normal.torch()
        
        # Check for any NaN or Inf values in the essential input vectors.
        valid_mask = ~torch.any(torch.isinf(pos) | torch.isnan(pos), dim=1)
        valid_mask &= ~torch.any(torch.isinf(normal) | torch.isnan(normal), dim=1)
        valid_mask &= ~torch.any(torch.isinf(wo) | torch.isnan(wo), dim=1)
        valid_mask &= ~torch.any(torch.isinf(wi) | torch.isnan(wi), dim=1)

        wi_len_sq = torch.sum(wi * wi, dim=1)
        valid_mask &= (wi_len_sq > 0.9) & (wi_len_sq < 1.1)

        # If NO valid data remains after filtering, abort the training step.
        if not torch.any(valid_mask):
            return (None,) * 6

        # Apply the final mask to all tensors to keep only the good data.
        pos = pos[valid_mask]
        wi = wi[valid_mask]
        wo = wo[valid_mask]
        normal = normal[valid_mask]
        targets_li = targets_li[valid_mask]

        roughness = torch.ones((pos.shape[0], 1), device=self.device)

        return pos, normal, wo, wi, roughness, targets_li

    def train_step(self, integrator) -> float:
        """Performs one training step for the radiance cache."""
        pos, normal, wo, wi, roughness, targets_li = self.prepare_training_data(integrator)

        if pos is None:
            return -1.0

        self.model.train()
        
        predicted_radiance = self.forward(pos, normal, wo, roughness)
        loss = torch.nn.functional.mse_loss(predicted_radiance, targets_li)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
