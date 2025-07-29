import torch
from guiding_network import GuidingNetwork
from vmf_mixture import BatchedMixedSphericalGaussianDistribution
from typing import Tuple

import drjit as dr
from math_utils import *

class PathGuidingSystem:
    def __init__(self, device: str = "cuda", K: int = 20, learning_rate: float = 2e-5):
        self.device = device
        self.gnn = GuidingNetwork(device, K=K).to(device)
        self.optimizer = torch.optim.Adam(self.gnn.model.parameters(), lr=learning_rate)
        print("Path Guiding System initialized.")

    def prepare_training_data(self, integrator) -> Tuple[torch.Tensor, ...]:
        """
        Processes the integrator's record to produce clean training tensors.
        """
        # This function is responsible for calling scatterDataIntoBuffer
        # and converting the record fields into PyTorch tensors.
        integrator.scatterDataIntoBuffer()
        rec = integrator.surfaceInteractionRecord

        if dr.width(rec.position) == 0:
            return None, None, None, None # Return None if no data

        pos = rec.position.torch()
        wo = rec.direction.torch() # This is the BSDF sample direction
        # For now, roughness is a placeholder
        roughness = torch.ones((pos.shape[0], 1), device=self.device)
        targets_li = rec.radiance.torch() # This is the incoming radiance Li
        bsdf_pdf = rec.bsdfPdf.torch() # Get the BSDF PDF for the sampled direction

        return pos, wo, roughness, targets_li, bsdf_pdf

    def train_step(self, integrator) -> float:
        """
        Performs a single epoch of training on the data from the integrator.
        
        Args:
            integrator: The PathGuidingIntegrator instance containing the latest path data.
            
        Returns:
            The calculated loss for this training step.
        """
        # Step 1: Prepare the data from the integrator
        pos, wo, roughness, targets_li, bsdf_pdf = self.prepare_training_data(integrator)

        if pos is None:
            print("Warning: Skipping training step due to no valid data.")
            return -1.0

        self.gnn.train()
        
        # Get the network's predicted distribution for all N samples
        predicted_params = self.gnn(pos, wo, roughness)
        predicted_dist = BatchedMixedSphericalGaussianDistribution(predicted_params)

        # Get the ground-truth directions and importance weights
        ground_truth_dirs = wo
        
        # Use Li as the target value
        ground_truth_radiance = targets_li.mean(axis=1)
        importance_clipping_threshold = 10.0
        ground_truth_radiance.clamp_(max=importance_clipping_threshold)


        # Calculate the importance-weighted loss
        prob = predicted_dist.pdf(ground_truth_dirs)
        epsilon = 1e-8

        importance_weight = ground_truth_radiance.detach() / (bsdf_pdf + epsilon)

        # 1. Main MLE Loss
        log_prob = torch.log(prob + epsilon)
        mle_loss = -importance_weight * log_prob

        loss = torch.mean(mle_loss)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gnn.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        
        loss_val = loss.item()
        if torch.isnan(loss):
             print("Error: Loss is NaN.")
             return -1.0
             
        return loss_val

    def pdf(self, position, wo, roughness):
        self.gnn.eval()
        with torch.no_grad():
            vmf_params = self.gnn(position, wo, roughness)
            guiding_dist = BatchedMixedSphericalGaussianDistribution(vmf_params)
            pdf_val = guiding_dist.pdf(wo)

        return pdf_val

    def sample_guided_direction(self, position, wo, roughness):
        """
        Uses the trained network to sample a new direction.
        """
        self.gnn.eval()
        with torch.no_grad():
            vmf_params = self.gnn(position, wo, roughness)
            
            guiding_dist = BatchedMixedSphericalGaussianDistribution(vmf_params)
            sampled_dirs = guiding_dist.sample()
            pdf_val = guiding_dist.pdf(sampled_dirs)
    
            return sampled_dirs, pdf_val