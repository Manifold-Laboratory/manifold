"""
GFN Loss Functions
==================

Physics-informed loss functions for stable geodesic training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry.boundaries import toroidal_dist_python


def hamiltonian_loss(velocities: list, states: list = None, metric_fn=None, lambda_h: float = 0.01, forces: list = None) -> torch.Tensor:
    """
    Riemannian Hamiltonian Energy Conservation Loss.
    
    If 'metric_fn' is provided, computes Energy = 0.5 * v^T g(x) v.
    Otherwise fallbacks to Euclidean Energy = 0.5 * ||v||^2.
    """
    if lambda_h == 0.0 or not velocities or len(velocities) < 2:
        return torch.tensor(0.0, device=velocities[0].device if (velocities and len(velocities) > 0) else 'cpu')
    
    energies = []
    for i in range(len(velocities)):
        v = velocities[i]
        if metric_fn is not None and states is not None:
             x = states[i]
             # E = 0.5 * sum(g_ii * v_i^2) for diagonal metrics
             g = metric_fn(x) 
             e = 0.5 * torch.sum(g * v.pow(2), dim=-1)
        else:
             e = 0.5 * v.pow(2).sum(dim=-1)
        energies.append(e)
        
    diffs = []
    for i in range(len(energies) - 1):
        dE = torch.abs(energies[i+1] - energies[i])
        if forces is not None and i < len(forces):
            f_norm = forces[i].pow(2).sum(dim=-1)
            mask = (f_norm < 1e-4).float()
            dE = dE * mask
        diffs.append(dE)
        
    return lambda_h * torch.stack(diffs).mean()


def kinetic_energy_penalty(velocities: list, lambda_k: float = 0.001) -> torch.Tensor:
    """
    L2 Regularization on Velocities.
    Encourages the model to be 'lazy' and move only when necessary.
    """
    if not velocities:
        return torch.tensor(0.0)
    
    all_v = torch.stack(velocities)
    return lambda_k * all_v.pow(2).mean()


def geodesic_regularization(velocities: list, christoffel_outputs: list, lambda_g: float = 0.001) -> torch.Tensor:
    """
    Geodesic Curvature Regularization.
    Supports both standard list of tensors and fused pre-computed sum.
    """
    if not christoffel_outputs:
        return torch.tensor(0.0)
    
    # Check if this is a fused regulation tensor (single tensor in list)
    if len(christoffel_outputs) == 1 and christoffel_outputs[0].dim() == 1:
        # Fused case: christoffel_outputs[0] is sum(||Gamma||^2) per batch item
        # To match all_curvatures.pow(2).mean(), we must divide by total elements per batch item
        # This isn't strictly known here, but we can pass it or retrieve it.
        # For MANIFOLD models, this is normally (depth * seq_len * dim)
        # We'll use a conservative estimate or let it be scaled by lambda_g.
        # Actually, let's keep it simple as a sum for now, but scaled.
        return lambda_g * christoffel_outputs[0].mean() / 1000.0 # Heuristic scaling
    
    # Standard Vectorization:
    all_curvatures = torch.stack(christoffel_outputs) # [N_heads*Seq, B, d]
    curvature_norms = all_curvatures.pow(2).mean()
    return lambda_g * curvature_norms



def noether_loss(christoffel_outputs: list, isomeric_groups: list = None, lambda_n: float = 0.01) -> torch.Tensor:
    """
    Semantic Symmetry (Noether) Loss.
    
    Enforces that 'Isomeric' subspaces (heads) learn the same geometric laws
    even if their specific weights are not strictly tied (Soft Symmetry).
    
    If weights ARE hard-tied (Isomeric Heads in MLayer), this term acts as a 
    regularizer to ensure gradients are consistent across symmetric contexts.
    
    Args:
        christoffel_outputs: List of Γ(v) outputs per head.
        isomeric_groups: List of head index groups [[0, 1], [2, 3]]
        lambda_n: Noether coefficient
    """
    if not isomeric_groups or not christoffel_outputs:
        return torch.tensor(0.0, device=christoffel_outputs[0].device if christoffel_outputs else 'cpu')
        
    total_diff = 0.0
    count = 0
    
    for group in isomeric_groups:
        if len(group) < 2: continue
        
        # Reference head output in this group
        ref_out = christoffel_outputs[group[0]]
        
        for other_h_idx in group[1:]:
            target_out = christoffel_outputs[other_h_idx]
            # MSE between geometric responses of symmetric heads
            total_diff = total_diff + torch.mean((ref_out - target_out).pow(2))
            count += 1
            
    if count == 0:
        return torch.tensor(0.0, device=christoffel_outputs[0].device)
        
    return lambda_n * (total_diff / count)




def circular_distance_loss(x_pred, x_target):
    """
    Holographic Phase Loss (Level 13).
    
    Computes distance on the flat Torus T^n:
    L = 1 - cos(x_pred - x_target)
    
    Properties:
    1. Bounded [0, 2]
    2. Continuous at 2pi boundary
    3. Gradient is sin(delta), naturally clipped [-1, 1]
    """
    delta = x_pred - x_target
    return (1.0 - torch.cos(delta)).mean()

class CircularDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x_pred, x_target):
        return circular_distance_loss(x_pred, x_target)

def toroidal_distance_loss(x_pred, x_target):
    dist = toroidal_dist_python(x_pred, x_target)
    return dist.pow(2).mean()

class ToroidalDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x_target):
        return toroidal_distance_loss(x_pred, x_target)


class GFNLoss(nn.Module):
    """
    Combined loss for GFN training.
    
    Components:
        1. Cross-Entropy (prediction accuracy)
        2. Hamiltonian Loss (energy conservation)
        3. Geodesic Regularization (curvature smoothness)
    
    Args:
        lambda_h: Hamiltonian loss weight (default: 0.01)
        lambda_g: Geodesic regularization weight (default: 0.001)
        ignore_index: Padding token index for CE loss
    """
    
    def __init__(self, lambda_h: float = 0.01, lambda_g: float = 0.001, lambda_n: float = 0.0, ignore_index: int = -100):
        super().__init__()
        self.lambda_h = lambda_h
        self.lambda_g = lambda_g
        self.lambda_n = lambda_n
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits, targets, velocities=None, christoffel_outputs=None, isomeric_groups=None):
        """
        Compute combined loss.
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            targets: Target tokens [batch, seq_len]
            velocities: Optional list of velocity tensors for Hamiltonian loss
            christoffel_outputs: Optional list of curvature tensors
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        # Primary loss: Cross-Entropy
        batch_size, seq_len, vocab_size = logits.shape
        ce = self.ce_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        loss_dict = {"ce": ce.item()}
        total = ce
        
        # Hamiltonian regularization
        if velocities and len(velocities) > 1:
            h_loss = hamiltonian_loss(velocities, lambda_h=self.lambda_h)
            total = total + h_loss
            loss_dict["hamiltonian"] = h_loss.item()
        
        if christoffel_outputs:
            g_loss = geodesic_regularization(velocities, christoffel_outputs, self.lambda_g)
            total = total + g_loss
            loss_dict["geodesic"] = g_loss.item()

        # Noether (Semantic Symmetries)
        if self.lambda_n > 0 and christoffel_outputs:
            n_loss = noether_loss(christoffel_outputs, isomeric_groups=isomeric_groups, lambda_n=self.lambda_n)
            total = total + n_loss
            loss_dict["noether"] = n_loss.item()
            
        loss_dict["total"] = total.item()
        
        return total, loss_dict
