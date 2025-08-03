import torch
import torch.optim.lr_scheduler as lr_scheduler 
from guiding_network import GuidingNetwork
from vmf_mixture import BatchedMixedSphericalGaussianDistribution
from typing import Tuple

import drjit as dr
from math_utils import *

class PathGuidingSystem:
    def __init__(self, device: str = "cuda", K: int = 20, learning_rate: float = 1e-2):
        self.device = device
        self.gnn = GuidingNetwork(device, K=K).to(device)
        self.optimizer = torch.optim.AdamW(self.gnn.model.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        print("Path Guiding System initialized.")

    def prepare_training_data(self, integrator) -> Tuple[torch.Tensor, ...]:
        """
        Processes the integrator's record to produce clean, validated training tensors.
        """
        rec = integrator.surfaceInteractionRecord
        integrator.scatter_data_into_buffer()

        if dr.width(rec.position) == 0:
            return (None,) * 6

        pos = rec.position.torch()
        wi = rec.wi.torch() # Direction of incoming ray
        wo = rec.wo.torch() # direction sampled using bsdf

        pos = 2 * ((pos - self.bbox_min)/(self.bbox_max - self.bbox_min)) - 1
        
        # print(pos.min(), pos.max())
        # print(self.bbox_min, self.bbox_max)

        targets_li = rec.radiance.torch()
        combined_pdf = rec.woPdf.torch()

        # BSDF PDF must be strictly positive.
        valid_mask = combined_pdf > 1e-8

        # Check for any NaN or Inf values in the essential input vectors.
        valid_mask &= ~torch.any(torch.isinf(pos) | torch.isnan(pos), dim=1)
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
        targets_li = targets_li[valid_mask]
        combined_pdf = combined_pdf[valid_mask]

        roughness = torch.ones((pos.shape[0], 1), device=self.device)

        return pos, wo, wi, roughness, targets_li, combined_pdf

    def train_step(self, integrator) -> float:
        # Prepare the data from the integrator
        pos, wo, wi, roughness, targets_li, combined_pdf = self.prepare_training_data(integrator)

        if pos is None:
            print("Warning: Skipping training step due to no valid data.")
            return -1.0

        self.gnn.train()
        
        # Get the network's predicted distribution for all N samples
        predicted_params = self.gnn(pos, wi, roughness)
        kappas = predicted_params['kappa']
        predicted_dist = BatchedMixedSphericalGaussianDistribution(predicted_params)
        
        # Use Li as the target value
        ground_truth_radiance = targets_li.mean(axis=1)
        importance_clipping_threshold = 10.0
        ground_truth_radiance.clamp_(max=importance_clipping_threshold)

        # Calculate the importance-weighted loss
        prob = predicted_dist.pdf(wo)
        epsilon = 1e-8

        importance_weight = ground_truth_radiance.detach() / (combined_pdf  + epsilon)
        importance_weight.clamp_(max=100)

        # Main MLE Importance weighted Loss
        log_prob = torch.log(prob + epsilon)
        mle_loss = -importance_weight * log_prob

        kappa_regularization_strength = 0.001
        kappa_loss = kappa_regularization_strength * torch.mean(kappas.pow(2))

        loss = torch.mean(mle_loss) + kappa_loss

        # Backpropagate
        if torch.isnan(loss) or torch.isinf(loss):
            print("\n--- Error: Loss is NaN or Inf. Dumping debug info. ---")
            
            # Check for NaNs in the inputs to the loss calculation
            print(f"NaN in 'prob': {torch.isnan(prob).any().item()}")
            print(f"NaN in 'ground_truth_radiance': {torch.isnan(ground_truth_radiance).any().item()}")
            print(f"NaN in 'combined_pdf ': {torch.isnan(combined_pdf ).any().item()}")
            print(f"NaN in 'importance_weight': {torch.isnan(importance_weight).any().item()}")
            
            # Check for zeros or negatives that could cause issues
            print(f"Min of 'prob': {torch.min(prob).item()}")
            print(f"Max of 'prob': {torch.max(prob).item()}")
            
            print(f"Min of 'combined_pdf ': {torch.min(combined_pdf ).item()}")
            print(f"Number of 'combined_pdf ' values near zero (<= {epsilon}): {torch.sum(combined_pdf  <= epsilon).item()}")

            # Check the values that go into the logarithm
            log_input = prob + epsilon
            print(f"Min of 'log_input' (prob + epsilon): {torch.min(log_input).item()}")
            print(f"Number of 'log_input' values <= 0: {torch.sum(log_input <= 0).item()}")
            
            print(f"Max of 'importance_weight': {torch.max(importance_weight).item()}")
            print("-----------------------------------------------------------------\n")

        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional: Check for NaN gradients
        for name, param in self.gnn.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"❗ NaN in gradient for {name}")

        torch.nn.utils.clip_grad_value_(self.gnn.model.parameters(), clip_value=1.0)
        torch.nn.utils.clip_grad_norm_(self.gnn.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        loss_val = loss.item()
     
        return loss_val

    def pdf(self, position, wi, roughness, wo):
        self.gnn.eval()

        with torch.no_grad():
            position = 2 * ((position - self.bbox_min)/(self.bbox_max - self.bbox_min))  - 1

            vmf_params = self.gnn(position, wi, roughness)
            guiding_dist = BatchedMixedSphericalGaussianDistribution(vmf_params)
            pdf_val = guiding_dist.pdf(wo)

        return pdf_val

    def sample_guided_direction(self, position, wi, roughness):
        """
        Uses the trained network to sample a new direction.
        """
        self.gnn.eval()

        with torch.no_grad():
            position = 2 * ((position - self.bbox_min)/(self.bbox_max - self.bbox_min)) - 1

            vmf_params = self.gnn.query(position, wi, roughness)
            
            guiding_dist = BatchedMixedSphericalGaussianDistribution(vmf_params)
            sampled_dirs = guiding_dist.sample()
            pdf_val = guiding_dist.pdf(sampled_dirs)
    
            return sampled_dirs, pdf_val