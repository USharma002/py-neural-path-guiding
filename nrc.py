import torch
import torch.nn as nn
import tinycudann as tcnn
import math
import torch.nn.functional as F

class NeuralRadianceCache(nn.Module):
    """
    A neural network using tiny-cuda-nn for Neural Radiance Cache
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Total input dimensions = 3 (pos) + 3 (normal) + 3 (diffuse color) + 3 (specular color) + 3 (dir) + 1 (roughness) = 16
        input_dims = 16
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