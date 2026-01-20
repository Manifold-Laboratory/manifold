"""
GFN Loss Functions
==================

Physics-informed loss functions for stable geodesic training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hamiltonian_loss(velocities: list, lambda_h: float = 0.01) -> torch.Tensor:
    """
    Hamiltonian Energy Conservation Loss.
    
    Penalizes the model if kinetic energy (||v||²) changes violently between 
    timesteps. This enforces smooth geodesic flow and prevents gradient explosion.
    
    Formula:
        L_H = λ * Σ_t |E_t - E_{t-1}|
        where E_t = ||v_t||²
    
    Args:
        velocities: List of velocity tensors [v_0, v_1, ..., v_T], each [batch, dim]
        lambda_h: Regularization strength (default: 0.01)
        
    Returns:
        Scalar loss tensor
    """
    if len(velocities) < 2:
        return torch.tensor(0.0, device=velocities[0].device)
    
    # Compute kinetic energy at each timestep: E = ||v||²
    energies = [v.pow(2).sum(dim=-1) for v in velocities]  # List of [batch]
    
    # Compute absolute energy differences
    energy_diffs = []
    for e1, e2 in zip(energies[:-1], energies[1:]):
        energy_diffs.append(torch.abs(e2 - e1))
    
    # Mean over time and batch
    total_diff = torch.stack(energy_diffs).mean()
    
    return lambda_h * total_diff


def geodesic_regularization(velocities: list, christoffel_outputs: list, lambda_g: float = 0.001) -> torch.Tensor:
    """
    Geodesic Curvature Regularization.
    
    Penalizes high curvature (large Christoffel outputs) to prevent 
    "semantic black holes" where gradients explode.
    
    Args:
        velocities: List of velocity tensors
        christoffel_outputs: List of Γ(v,v) outputs from Christoffel networks
        lambda_g: Regularization strength
        
    Returns:
        Scalar loss tensor
    """
    if not christoffel_outputs:
        return torch.tensor(0.0)
    
    # Efficient Vectorization: Instead of 480+ CUDA kernels, we do one.
    # christoffel_outputs is a list of [batch, d] tensors.
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
            h_loss = hamiltonian_loss(velocities, self.lambda_h)
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
