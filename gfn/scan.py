import torch
import math

# Try to import CUDA kernels
try:
    from .cuda import cuda_ops
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

def parallel_scan(a, x):
    """
    Computes y_t = a_t * y_{t-1} + x_t via parallel associative scan.
    
    Args:
        a: Multiplicative term [batch, seq_len, dim]
        x: Additive term [batch, seq_len, dim]
        
    Returns:
        y: Scan result [batch, seq_len, dim]
    """
    # Use CUDA kernel if available and on GPU
    if CUDA_AVAILABLE and a.is_cuda:
        return cuda_ops.parallel_scan_fused(a, x)
    
    # Fallback to PyTorch implementation
    # Simply using PyTorch's native cumulatives is often faster/stable enough 
    # for sequences < 4096 than custom cuda kernels without Triton.
    # But native cumprod is unstable for deep recurrences.
    # We implement a parallel prefix scan using log-space reduction if needed,
    # or the standard recursive doubling algorithm.
    
    # For MANIFOLD, we are solving v_t = decay_t * v_{t-1} + force_t
    
    # 1. Compute cumulative product of 'a' (decays)
    # log_a = torch.log(a.clamp(min=1e-6))
    # cum_log_a = torch.cumsum(log_a, dim=1)
    # cum_a = torch.exp(cum_log_a)
    
    # This is numerically unstable if 'a' > 1.
    # For now, we utilize a sequential fallback for correctness or a 
    # weak parallel implementation. 
    # REAL IMPLEMENTATION: We will use the "Blelloc Scan" algorithm in pure PyTorch.
    
    # Algorithm:
    # y_t = x_t + a_t * y_{t-1}
    # This is a first-order linear recurrence.
    
    # Efficient PyTorch implementation of parallel scan is non-trivial without 
    # custom kernels (like selective_scan_cuda). 
    # However, for prototyping "Proof of Concept" of scan support,
    # we can use a recursive doubling approach.
    
    B, L, D = x.shape
    
    if L < 32:
        # Sequential is faster for tiny/short sequences
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, device=x.device)
        for t in range(L):
            h = a[:, t] * h + x[:, t]
            y[:, t] = h
        return y

    # Recursive Doubling (Hillis-Steele) - O(log N) depth, O(N log N) work
    # This is NOT work-efficient but is parallel depth efficient.
    # Good for modern GPUs with massive parallelism.
    
    curr_a = a.clone()
    curr_x = x.clone()
    
    # Calculate number of steps: log2(L)
    steps = int(math.ceil(math.log2(L)))
    
    for i in range(steps):
        shift = 2**i
        
        # Shifted values
        prev_a = torch.roll(curr_a, shifts=shift, dims=1)
        prev_x = torch.roll(curr_x, shifts=shift, dims=1)
        
        # Mask out wrapped around elements
        # (Technically we can zero them, but simpler logic:)
        # We only want to combine with elements strictly 'before' us in time.
        # torch.roll wraps around, so indices [0, ... shift-1] get values from end.
        # We must mask them.
        
        mask = torch.ones(L, device=x.device)
        mask[:shift] = 0
        mask = mask.view(1, L, 1)
        
        # Matrix multiplication in log-space/linear space of the update operator
        # New operator composition:
        # (a2, x2) o (a1, x1) = (a2*a1, a2*x1 + x2)
        
        # Update
        new_a = curr_a * prev_a
        new_x = curr_a * prev_x + curr_x
        
        # Apply mask: for t < shift, we don't change
        curr_a = torch.where(mask > 0.5, new_a, curr_a)
        curr_x = torch.where(mask > 0.5, new_x, curr_x)
        
    return curr_x

class ParallelScan(torch.autograd.Function):
    """
    A custom autograd function could be used here for memory efficiency,
    but we start with the pure PyTorch impl above.
    """
    pass
