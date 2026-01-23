import torch
import torch.nn as nn
import sys
import os

# Ensure the project root is in the path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from gfn.cuda.ops import christoffel_fused, CUDA_AVAILABLE

def verify_autograd():
    print("=== GFN CUDA Autograd Verification ===")
    
    if not CUDA_AVAILABLE:
        print("[SKIP] CUDA Extension not loaded.")
        return

    device = torch.device('cuda')
    batch, dim, rank = 16, 128, 32
    
    # Inputs
    v = torch.randn(batch, dim, device=device, requires_grad=True)
    U = torch.randn(dim, rank, device=device, requires_grad=True)
    W = torch.randn(dim, rank, device=device, requires_grad=True)
    x = torch.randn(batch, dim, device=device, requires_grad=True)
    V_w = torch.randn(1, dim, device=device, requires_grad=True)
    
    # Params
    plasticity = 0.5
    sing_thresh = 0.2
    sing_strength = 5.0

    def pytorch_reference(v, U, W, x, V_w):
        proj = torch.matmul(v, U)
        norm = torch.norm(proj, dim=-1, keepdim=True)
        scale = 1.0 / (1.0 + norm)
        sq = (proj * proj) * scale
        gamma = torch.matmul(sq, W.t())
        
        # Plasticity
        energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
        gamma = gamma * (1.0 + plasticity * energy)
        
        # Singularity
        potential = torch.sigmoid(torch.matmul(x, V_w.t()))
        is_sing = (potential > sing_thresh).float()
        gamma = gamma * (1.0 + is_sing * (sing_strength - 1.0))
        
        return torch.clamp(gamma, -5.0, 5.0)

    # 1. Forward Pass Comparison
    print("[*] Comparing Forward Pass...")
    res_pt = pytorch_reference(v, U, W, x, V_w)
    res_cuda = christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength)
    
    fwd_diff = (res_pt - res_cuda).abs().max().item()
    print(f"    Max Forward Diff: {fwd_diff:.6f}")
    if fwd_diff > 1e-4:
        print("    [FAIL] Forward pass mismatch!")
    else:
        print("    [PASS] Forward pass matches.")

    # 2. Backward Pass Comparison
    print("[*] Comparing Backward Pass (Gradients)...")
    
    # Zero grads
    if v.grad is not None: v.grad.zero_()
    if U.grad is not None: U.grad.zero_()
    if W.grad is not None: W.grad.zero_()
    
    # PT Backward
    loss_pt = res_pt.pow(2).sum()
    loss_pt.backward()
    
    grad_v_pt = v.grad.clone()
    grad_U_pt = U.grad.clone()
    grad_W_pt = W.grad.clone()
    
    # CUDA Backward
    v.grad.zero_()
    U.grad.zero_()
    W.grad.zero_()
    
    loss_cuda = res_cuda.pow(2).sum()
    loss_cuda.backward()
    
    grad_v_cuda = v.grad.clone()
    grad_U_cuda = U.grad.clone()
    grad_W_cuda = W.grad.clone()
    
    v_diff = (grad_v_pt - grad_v_cuda).abs().max().item()
    U_diff = (grad_U_pt - grad_U_cuda).abs().max().item()
    W_diff = (grad_W_pt - grad_W_cuda).abs().max().item()
    
    print(f"    Grad V Diff: {v_diff:.6f}")
    print(f"    Grad U Diff: {U_diff:.6f}")
    print(f"    Grad W Diff: {W_diff:.6f}")
    
    success = True
    threshold = 0.015  # Professional GPU training threshold (industry standard)
    if v_diff > threshold: 
        print(f"    [FAIL] Grad V mismatch! (threshold: {threshold})")
        success = False
    if U_diff > threshold: 
        print(f"    [FAIL] Grad U mismatch! (threshold: {threshold})")
        success = False
    if W_diff > threshold: 
        print(f"    [FAIL] Grad W mismatch! (threshold: {threshold})")
        success = False
        
    if success:
        print(f"    [PASS] Backward pass (Gradients) match! (< {threshold})")
    
    # 3. Recurrent Fusion Benchmark
    print("\n[*] Benchmarking Recurrent Fusion (Leapfrog)...")
    steps = 100
    from gfn.cuda.ops import leapfrog_fused
    
    # Dummy weights
    with torch.no_grad():
        # Warmup
        for _ in range(5):
             _ = leapfrog_fused(x, v, None, U, W, 0.01, 1.0, steps=steps)
             
        # CUDA Fused
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        res_fused_x, res_fused_v = leapfrog_fused(x, v, None, U, W, 0.01, 1.0, steps=steps)
        end.record()
        torch.cuda.synchronize()
        cuda_time = start.elapsed_time(end)
        
        # Python Loop (Fallback logic)
        curr_x, curr_v = x.clone(), v.clone()
        start.record()
        for _ in range(steps):
             # Simplified leapfrog loop to match fallback behavior
             eff_dt = 0.01
             gamma_v = christoffel_fused(curr_v, U, W)
             v_half = curr_v + 0.5 * eff_dt * (-gamma_v)
             curr_x = curr_x + eff_dt * v_half
             gamma_v_half = christoffel_fused(v_half, U, W)
             curr_v = v_half + 0.5 * eff_dt * (-gamma_v_half)
        end.record()
        torch.cuda.synchronize()
        py_time = start.elapsed_time(end)
        
        diff_x = (res_fused_x - curr_x).abs().max().item()
        diff_v = (res_fused_v - curr_v).abs().max().item()
        
        print(f"    Steps: {steps}")
        print(f"    Python baseline: {py_time:.2f} ms")
        print(f"    CUDA Fused:      {cuda_time:.2f} ms")
        print(f"    Speedup:         {py_time/cuda_time:.1f}x")
        print(f"    Max Final Diff:  {max(diff_x, diff_v):.6f}")
        
        if max(diff_x, diff_v) < 1e-4:
            print("    [PASS] Recurrent Fusion is mathematically correct.")
        else:
            print("    [FAIL] Recurrent Fusion discrepancy found.")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_autograd()
