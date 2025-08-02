import torch
import torch.nn as nn
import tinycudann as tcnn
import math
import torch.nn.functional as F

class NeuralRadianceCache(nn.Module):
    """
    A neural network using tiny-cuda-nn to predict parameters for a 
    von Mises-Fisher (vMF) mixture model.
    
    The network takes a 7D input (position, view direction, roughness) and
    outputs the parameters for a mixture of K vMF distributions.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Total input dimensions = 3 (pos) + 3 (dir) + 1 (roughness) = 7
        input_dims = 7
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
                    },
                    {
                        "n_dims_to_encode": 4,
                        "otype": "Identity",
                    }
                ]
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
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

    def _prepare_input(self, positions, view_dirs, roughness):
        """Helper to concatenate and validate inputs."""
        # A more robust implementation would normalize position based on the scene AABB
        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)

        positions = (positions + 1.0) / 2.0
        view_dirs = (view_dirs + 1.0) / 2.0

        input_tensor = torch.cat([positions, view_dirs, roughness], dim=1)
        return input_tensor

    def process_output(self, raw_output: torch.Tensor) -> torch.Tensor:
        """
        Applies a final activation to the raw network output to ensure
        radiance is non-negative.
        """
        # Using softplus is a good choice. It's like a smooth ReLU, ensuring positivity.
        # Adding a small epsilon can improve numerical stability if the output is used in a log.
        return F.softplus(raw_output)

    def query(self, positions, view_dirs, roughness):
        """
        Queries the network and returns processed vMF mixture parameters.
        """
        # Set to evaluation mode and disable gradients for inference speed
        self.model.eval()

        with torch.no_grad():
            input_tensor = self._prepare_input(positions, view_dirs, roughness)
            # The network expects inputs in the [0, 1] range.
            # Normalizing with the scene AABB is the proper way to do this.
            # A simple (pos + 1)/2 is a placeholder for a [-1, 1] world.
            raw_output = self.model(input_tensor.to(self.device))
        
        # Process the raw output to get meaningful parameters
        return self.process_output(raw_output)

    def forward(self, positions, view_dirs, roughness):
        """
        A forward pass for TRAINING. Gradients are tracked.
        """
        # Note: No model.eval() and no torch.no_grad()
        input_tensor = self._prepare_input(positions, view_dirs, roughness)

        # The tcnn model handles the device placement
        raw_output = self.model(input_tensor.contiguous())
        
        self.process_output(raw_output)