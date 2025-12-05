"""Practical Path Guiding (PPG) - Robust PyTorch Implementation.

Reference: Müller et al., "Practical Path Guiding", HPG 2017.

Implementation Details:
- Spatial Tree: Binary k-D Tree with SAH-like (spatial median) splitting.
- Directional Tree: Quadtree with Cylindrical Equal-Area Mapping.
- Robustness: Handles zero-flux, dynamic memory resizing, and batch sparsity.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import math

from guiding.base import GuidingDistribution, GuidingConfig
from guiding.training_data import TrainingBatch
from guiding.config import get_logger

logger = get_logger("ppg_guiding")

@dataclass
class PPGConfig(GuidingConfig):
    # Spatial Tree (Binary Tree)
    kdtree_max_depth: int = 20         
    spatial_split_threshold: float = 0.01  # Split node if > 1% of total flux
    
    # Directional Tree (Quadtree)
    quadtree_max_depth: int = 20       
    quadtree_flux_threshold: float = 0.01  # Relative flux threshold for splitting
    
    # Scene bounds
    bbox_min: Optional[Tuple[float, float, float]] = None
    bbox_max: Optional[Tuple[float, float, float]] = None
    
    # Training Dynamics
    training_beta: float = 0.5   # 0.5 = Retain 50% history, Learn 50% new
    refinement_iterations: int = 1


class DirectionalQuadTreeForest:
    """
    Represents a forest of Quadtrees. Each spatial leaf points to one root in this forest.
    Uses Cylindrical Equal-Area Mapping.
    """
    def __init__(self, max_depth: int = 20, initial_capacity: int = 500000, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_depth = max_depth
        
        # Node Layout: [Flux, Child_Start_Idx, Depth, Unused]
        # Child_Start_Idx: -1 indicates a Leaf Node. 
        # If not leaf, children are at indices: [idx, idx+1, idx+2, idx+3]
        self.capacity = initial_capacity
        self.nodes = torch.zeros((self.capacity, 4), device=self.device)
        self.next_free_idx = 0
        
        # Mapping: Spatial Leaf ID -> D-Tree Root Node Index
        self.roots = torch.full((32768,), -1, dtype=torch.long, device=self.device)

    def _ensure_roots_size(self, spatial_idx_max: int):
        if spatial_idx_max >= self.roots.shape[0]:
            new_size = max(spatial_idx_max + 1, self.roots.shape[0] * 2)
            new_roots = torch.full((new_size,), -1, dtype=torch.long, device=self.device)
            new_roots[:self.roots.shape[0]] = self.roots
            self.roots = new_roots

    def _ensure_nodes_capacity(self, count: int):
        if self.next_free_idx + count >= self.capacity:
            new_cap = max(self.capacity * 2, self.capacity + count + 100000)
            new_nodes = torch.zeros((new_cap, 4), device=self.device)
            new_nodes[:self.capacity] = self.nodes
            self.nodes = new_nodes
            self.capacity = new_cap

    def allocate_root(self, spatial_leaf_idx: int) -> int:
        self._ensure_roots_size(spatial_leaf_idx)
        self._ensure_nodes_capacity(1)
        
        idx = self.next_free_idx
        self.next_free_idx += 1
        
        # Init with epsilon flux to prevent division by zero during early sampling
        self.nodes[idx, 0] = 1e-6 
        self.nodes[idx, 1] = -1  # Leaf
        self.nodes[idx, 2] = 0   # Depth 0
        
        self.roots[spatial_leaf_idx] = idx
        return idx

    def _allocate_nodes(self, count: int) -> int:
        self._ensure_nodes_capacity(count)
        start_idx = self.next_free_idx
        self.next_free_idx += count
        return start_idx

    def accumulate(self, spatial_leaf_indices: torch.Tensor, uv: torch.Tensor, flux: torch.Tensor):
        if len(spatial_leaf_indices) == 0: return
        
        # Ensure UVs are strictly in [0, 1) to avoid index wrapping
        uv = uv.clamp(0.0, 0.999999)
        
        if spatial_leaf_indices.max() >= self.roots.shape[0]:
             self._ensure_roots_size(spatial_leaf_indices.max().item())
            
        root_indices = self.roots[spatial_leaf_indices]
        valid_mask = root_indices >= 0
        if not valid_mask.any(): return
        
        active_nodes = root_indices[valid_mask]
        active_uv = uv[valid_mask]
        active_flux = flux[valid_mask]

        # Traverse and deposit flux
        for depth in range(self.max_depth + 1):
            # 1. Deposit Flux (Atomic Add)
            self.nodes[:, 0].scatter_add_(0, active_nodes, active_flux)
            
            # 2. Check Leaf Status
            child_base_idx = self.nodes[active_nodes, 1].long()
            is_leaf = child_base_idx == -1
            if is_leaf.all(): break
                
            non_leaf_mask = ~is_leaf
            if not non_leaf_mask.any(): break
            
            # 3. Filter Active Paths
            active_nodes = active_nodes[non_leaf_mask]
            active_uv = active_uv[non_leaf_mask]
            active_flux = active_flux[non_leaf_mask]
            base_indices = child_base_idx[non_leaf_mask]
            
            # 4. Select Child (Morton Code / Z-Order Curve)
            scaler = 2.0 ** (depth + 1)
            u_check = (active_uv[:, 0] * scaler).long() % 2
            v_check = (active_uv[:, 1] * scaler).long() % 2
            child_offset = u_check + (v_check * 2)
            
            active_nodes = base_indices + child_offset

    def decay(self, factor: float):
        """Exponential Moving Average decay for temporal stability."""
        if self.next_free_idx > 0:
            self.nodes[:self.next_free_idx, 0] *= factor

    def refine(self, threshold_ratio: float = 0.01):
        active_roots = self.roots[self.roots != -1]
        if len(active_roots) == 0: return
        
        # Threshold is relative to the root flux of each tree
        root_fluxes = self.nodes[active_roots, 0]
        
        curr_nodes = active_roots
        curr_thresholds = root_fluxes * threshold_ratio
        
        # Clamp min threshold to prevent infinite splitting on noise
        curr_thresholds = curr_thresholds.clamp(min=1e-5)
        
        for d in range(self.max_depth):
            is_leaf = self.nodes[curr_nodes, 1] == -1
            should_split = (self.nodes[curr_nodes, 0] > curr_thresholds) & is_leaf
            
            if should_split.any():
                split_indices = curr_nodes[should_split]
                num_splits = len(split_indices)
                
                # Allocate blocks of 4 nodes
                new_start_idx = self._allocate_nodes(num_splits * 4)
                
                # Link parent to children
                child_starts = torch.arange(0, num_splits * 4, 4, device=self.device) + new_start_idx
                self.nodes[split_indices, 1] = child_starts.float()
                
                # Flux Inheritance: Distribute parent flux to children
                parent_flux = self.nodes[split_indices, 0]
                child_flux = parent_flux / 4.0
                
                for i in range(4):
                    c_idx = child_starts + i
                    self.nodes[c_idx, 0] = child_flux
                    self.nodes[c_idx, 1] = -1 
                    self.nodes[c_idx, 2] = d + 1
            
            # Prepare next depth
            not_leaf = self.nodes[curr_nodes, 1] != -1
            if not not_leaf.any(): break
                
            parents = curr_nodes[not_leaf]
            parent_thresholds = curr_thresholds[not_leaf]
            first_child = self.nodes[parents, 1].long()
            
            next_nodes_list = []
            next_thresh_list = []
            for i in range(4):
                next_nodes_list.append(first_child + i)
                next_thresh_list.append(parent_thresholds)
                
            curr_nodes = torch.cat(next_nodes_list)
            curr_thresholds = torch.cat(next_thresh_list)

    def sample(self, spatial_leaf_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = len(spatial_leaf_indices)
        if N == 0:
            return torch.zeros((0,2), device=self.device), torch.zeros((0,), device=self.device)

        if spatial_leaf_indices.max() >= self.roots.shape[0]:
             return torch.zeros((N,2), device=self.device), torch.zeros((N,), device=self.device)

        active_nodes = self.roots[spatial_leaf_indices]
        
        uv_min = torch.zeros((N, 2), device=self.device)
        uv_scale = torch.ones((N, 2), device=self.device)
        pdf = torch.ones(N, device=self.device)
        
        xi = torch.rand((N, 2), device=self.device)
        
        for _ in range(self.max_depth):
            child_base = self.nodes[active_nodes, 1].long()
            is_leaf = child_base == -1
            if is_leaf.all(): break
                
            mask = ~is_leaf
            if not mask.any(): break
            base = child_base[mask]
            
            # Fetch 4 children fluxes
            c0 = self.nodes[base, 0]
            c1 = self.nodes[base + 1, 0]
            c2 = self.nodes[base + 2, 0]
            c3 = self.nodes[base + 3, 0]
            fluxes = torch.stack([c0, c1, c2, c3], dim=1)
            
            # Robust Probability Calculation
            total_flux = fluxes.sum(dim=1, keepdim=True)
            
            # Uniform fallback for zero flux regions
            probs = torch.full_like(fluxes, 0.25)
            has_flux = total_flux.squeeze(1) > 1e-9
            if has_flux.any():
                probs[has_flux] = fluxes[has_flux] / total_flux[has_flux]
            
            # Inverse CDF Sampling
            cdf = torch.cumsum(probs, dim=1)
            cdf[:, 3] = 1.0 # Enforce exact 1.0
            
            rand_val = xi[mask, 0]
            selection = (rand_val.unsqueeze(1) > cdf[:, :3]).sum(dim=1)
            
            selected_prob = probs.gather(1, selection.unsqueeze(1)).squeeze(1)
            pdf[mask] *= selected_prob * 4.0 # Jacobian scaling for area reduction
            
            active_nodes[mask] = base + selection
            
            # Hierarchical Sample Warping (Rescale Random Variable)
            cdf_low = torch.zeros_like(rand_val)
            has_prev = selection > 0
            if has_prev.any():
                cdf_low[has_prev] = cdf.gather(1, (selection[has_prev]-1).unsqueeze(1)).squeeze(1)
            
            cdf_high = cdf.gather(1, selection.unsqueeze(1)).squeeze(1)
            denom = (cdf_high - cdf_low).clamp(min=1e-8)
            xi[mask, 0] = (rand_val - cdf_low) / denom
            
            u_offset = (selection % 2).float() 
            v_offset = (selection // 2).float()
            
            uv_scale[mask] *= 0.5
            uv_min[mask, 0] += u_offset * uv_scale[mask, 0]
            uv_min[mask, 1] += v_offset * uv_scale[mask, 1]
            
        final_uv = uv_min + xi * uv_scale
        return final_uv, pdf
        
    def pdf(self, spatial_leaf_indices: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        N = len(spatial_leaf_indices)
        if N == 0: return torch.zeros((0,), device=self.device)
        
        uv = uv.clamp(0.0, 0.999999)

        if spatial_leaf_indices.max() >= self.roots.shape[0]:
             return torch.zeros(N, device=self.device)

        active_nodes = self.roots[spatial_leaf_indices]
        pdf = torch.ones(N, device=self.device)
        
        for depth in range(self.max_depth):
            child_base = self.nodes[active_nodes, 1].long()
            is_leaf = child_base == -1
            if is_leaf.all(): break
            
            mask = ~is_leaf
            if not mask.any(): break
            base = child_base[mask]
            uv_m = uv[mask]
            
            c0 = self.nodes[base, 0]
            c1 = self.nodes[base+1, 0]
            c2 = self.nodes[base+2, 0]
            c3 = self.nodes[base+3, 0]
            fluxes = torch.stack([c0, c1, c2, c3], dim=1)
            total = fluxes.sum(1, keepdim=True)
            
            probs = torch.full_like(fluxes, 0.25)
            has_flux = total.squeeze(1) > 1e-9
            if has_flux.any():
                probs[has_flux] = fluxes[has_flux] / total[has_flux]
            
            scaler = 2.0 ** (depth + 1)
            u_bit = (uv_m[:, 0] * scaler).long() % 2
            v_bit = (uv_m[:, 1] * scaler).long() % 2
            selection = u_bit + v_bit * 2
            
            sel_prob = probs.gather(1, selection.unsqueeze(1)).squeeze(1)
            pdf[mask] *= sel_prob * 4.0
            
            active_nodes[mask] = base + selection
            
        return pdf


class SpatialBinaryTree:
    """
    Binary K-D Tree for spatial partitioning.
    Stores bounds and points to Directional Trees in leaf nodes.
    """
    def __init__(self, bbox_min, bbox_max, max_depth=20, device=None):
        self.device = device
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.max_depth = max_depth
        
        # Node Layout: [Axis, SplitVal, LeftIdx, RightIdx, Flux, DTreeIdx]
        # Axis: -1 = Leaf, 0=X, 1=Y, 2=Z
        self.capacity = 100000 
        self.nodes = torch.zeros((self.capacity, 6), device=self.device)
        
        # Init Root
        self.nodes[0, 0] = -1 
        self.nodes[0, 5] = 0 # Points to DTree Root 0
        self.node_count = 1
        
        # AABB Cache
        self.node_mins = torch.zeros((self.capacity, 3), device=self.device)
        self.node_maxs = torch.zeros((self.capacity, 3), device=self.device)
        self.node_mins[0] = bbox_min
        self.node_maxs[0] = bbox_max

    def _ensure_capacity(self, count=1):
        if self.node_count + count >= self.nodes.shape[0]:
            new_cap = max(self.nodes.shape[0] * 2, self.node_count + count + 10000)
            
            new_nodes = torch.zeros((new_cap, 6), device=self.device)
            new_nodes[:self.nodes.shape[0]] = self.nodes
            self.nodes = new_nodes
            
            new_mins = torch.zeros((new_cap, 3), device=self.device)
            new_mins[:self.node_mins.shape[0]] = self.node_mins
            self.node_mins = new_mins
            
            new_maxs = torch.zeros((new_cap, 3), device=self.device)
            new_maxs[:self.node_maxs.shape[0]] = self.node_maxs
            self.node_maxs = new_maxs
            
            logger.info(f"Resized Spatial Tree to {new_cap}")

    def get_leaf_indices(self, positions: torch.Tensor) -> torch.Tensor:
        active_indices = torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
        
        for _ in range(self.max_depth + 2):
            axes = self.nodes[active_indices, 0].long()
            is_leaf = axes == -1
            if is_leaf.all(): break
            
            mask = ~is_leaf
            if not mask.any(): break
            
            curr = active_indices[mask]
            ax = axes[mask]
            split_vals = self.nodes[curr, 1]
            
            # Explicit branching logic
            left_child = self.nodes[curr, 2].long()
            right_child = self.nodes[curr, 3].long()
            
            pos_vals = positions[mask].gather(1, ax.unsqueeze(1)).squeeze(1)
            go_right = pos_vals >= split_vals
            
            next_indices = torch.where(go_right, right_child, left_child)
            active_indices[mask] = next_indices
            
        return active_indices

    def refine(self, dtree_forest: DirectionalQuadTreeForest, threshold_flux: float):
        all_indices = torch.arange(self.node_count, device=self.device)
        is_leaf = self.nodes[:self.node_count, 0] == -1
        
        # Split criteria: Leaf node with flux > 1% of total
        high_flux = self.nodes[:self.node_count, 4] > threshold_flux
        candidates = all_indices[is_leaf & high_flux]
        
        for idx in candidates:
            self._ensure_capacity(2)
                
            b_min = self.node_mins[idx]
            b_max = self.node_maxs[idx]
            extent = b_max - b_min
            axis = torch.argmax(extent).item()
            split_pos = (b_min[axis] + b_max[axis]) / 2.0
            
            left_idx = self.node_count
            right_idx = self.node_count + 1
            self.node_count += 2
            
            # Configure Parent
            self.nodes[idx, 0] = axis
            self.nodes[idx, 1] = split_pos
            self.nodes[idx, 2] = left_idx
            self.nodes[idx, 3] = right_idx
            
            # Configure Children
            for c_idx, is_right in [(left_idx, False), (right_idx, True)]:
                self.nodes[c_idx, 0] = -1 
                self.nodes[c_idx, 4] = 0.0 # Reset flux for new iteration
                self.nodes[c_idx, 5] = 0   
                
                c_min = b_min.clone()
                c_max = b_max.clone()
                if is_right: c_min[axis] = split_pos
                else: c_max[axis] = split_pos
                self.node_mins[c_idx] = c_min
                self.node_maxs[c_idx] = c_max
                
                # Allocate new directional tree for child
                dtree_idx = dtree_forest.allocate_root(c_idx)
                self.nodes[c_idx, 5] = dtree_idx


class PPGGuidingDistribution(GuidingDistribution):
    def __init__(self, config: Optional[PPGConfig] = None):
        config = config or PPGConfig()
        super().__init__(config)
        self.ppg_config = config
        self.stree = None
        self.dtree_forest = None
        self.iteration = 0
        
    @property
    def name(self) -> str: return "PPG (Pytorch)"

    def _init_structures(self):
        if self.stree is None:
            bbox_min = torch.tensor(self.ppg_config.bbox_min or [-100.0]*3, device=self.device)
            bbox_max = torch.tensor(self.ppg_config.bbox_max or [100.0]*3, device=self.device)
            self.stree = SpatialBinaryTree(bbox_min, bbox_max, self.ppg_config.kdtree_max_depth, self.device)
            self.dtree_forest = DirectionalQuadTreeForest(self.ppg_config.quadtree_max_depth, device=self.device)
            self.dtree_forest.allocate_root(0)

    def set_scene_bounds(self, bbox_min: torch.Tensor, bbox_max: torch.Tensor):
        self.ppg_config.bbox_min = tuple(bbox_min.cpu().numpy().tolist())
        self.ppg_config.bbox_max = tuple(bbox_max.cpu().numpy().tolist())
        self.stree = None
        self._init_structures()

    def _dir_to_uv(self, d: torch.Tensor) -> torch.Tensor:
        # Cylindrical Equal-Area Mapping (Jacobian = 1/4pi)
        # v maps z: [-1, 1] -> [0, 1]
        v = (d[:, 2] + 1.0) * 0.5
        # u maps phi: [-pi, pi] -> [0, 1]
        phi = torch.atan2(d[:, 1], d[:, 0])
        u = (phi + math.pi) * (0.5 / math.pi)
        return torch.stack([u, v], dim=1)

    def _uv_to_dir(self, uv: torch.Tensor) -> torch.Tensor:
        u, v = uv[:, 0], uv[:, 1]
        z = 2.0 * v - 1.0
        phi = u * (2.0 * math.pi) - math.pi
        sin_theta = torch.sqrt((1.0 - z*z).clamp(min=0.0))
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        return torch.stack([x, y, z], dim=1)

    def sample(self, position, wi, roughness):
        self._init_structures()
        leaf_indices = self.stree.get_leaf_indices(position)
        uv, pdf_uv = self.dtree_forest.sample(leaf_indices)
        wo = self._uv_to_dir(uv)
        # PDF Change of Variables: p(w) = p(uv) / |d(w)/d(uv)|
        # For Cylindrical Equal Area: |d(w)/d(uv)| = 4pi
        pdf_w = pdf_uv * (1.0 / (4.0 * math.pi))
        return wo, pdf_w

    def pdf(self, position, wi, wo, roughness):
        self._init_structures()
        leaf_indices = self.stree.get_leaf_indices(position)
        uv = self._dir_to_uv(wo)
        pdf_uv = self.dtree_forest.pdf(leaf_indices, uv)
        pdf_w = pdf_uv * (1.0 / (4.0 * math.pi))
        return pdf_w
    
    def log_pdf(self, position, wi, wo, roughness):
        p = self.pdf(position, wi, wo, roughness)
        return torch.log(p.clamp(min=1e-10))

    def train_step(self, batch: TrainingBatch) -> float:
        self._init_structures()
        
        pos = batch.position
        wo = batch.wo
        radiance = batch.radiance_rgb.mean(dim=1)
        pdf_orig = batch.combined_pdf
        
        # Estimate Flux: Phi = L / p
        flux = radiance / pdf_orig.clamp(min=1e-8)
        flux = flux.clamp(max=1e5) # Clamping prevents fireflies from destroying the tree
        
        # 1. Accumulate Flux
        leaf_indices = self.stree.get_leaf_indices(pos)
        self.stree.nodes[:self.stree.node_count, 4].scatter_add_(0, leaf_indices, flux)
        
        uv = self._dir_to_uv(wo)
        self.dtree_forest.accumulate(leaf_indices, uv, flux)
        
        # 2. Refine & Decay (Periodic)
        self.iteration += 1
        if self.iteration % self.ppg_config.refinement_iterations == 0:
            logger.info(f"Refining PPG Iteration {self.iteration}")
            
            # a. Refine Directional Trees (Quadtrees)
            self.dtree_forest.refine(self.ppg_config.quadtree_flux_threshold)
            
            # b. Refine Spatial Tree (Binary Tree)
            self.stree.refine(self.dtree_forest, self.ppg_config.spatial_split_threshold)
            
            # c. Decay History (Temporal Averaging)
            self.dtree_forest.decay(self.ppg_config.training_beta)
            self.stree.nodes[:, 4] *= self.ppg_config.training_beta
        
        # Return variance for logging
        weights = radiance / pdf_orig.clamp(min=1e-8)
        return torch.var(weights).item()

    def get_distribution_for_visualization(self, position, wi, roughness):
        return self

    def reset_tree_data(self):
        if self.dtree_forest: self.dtree_forest.nodes[:, 0].zero_()
        if self.stree: self.stree.nodes[:, 4].zero_()

    def get_tree_stats(self) -> dict:
        if not self.stree: return {}
        return {
            "spatial_nodes": int(self.stree.node_count),
            "directional_nodes": int(self.dtree_forest.next_free_idx),
            "iteration": self.iteration
        }