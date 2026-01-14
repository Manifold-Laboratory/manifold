
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cuda.ops import christoffel_fused, leapfrog_fused, CUDA_AVAILABLE

def verify_kernels():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GFN Custom Kernels Loaded: {CUDA_AVAILABLE}")
    
    if not CUDA_AVAILABLE:
        print("Skipping verification as custom kernels are not loaded.")
        return

    device = torch.device('cuda')
    torch.manual_seed(42)

    # --- Verify Christoffel Fused ---
    print("\n--- Verifying Christoffel Fused ---")
    batch_size = 32
    dim = 64
    rank = 16
    
    # Scale inputs down to avoid clamp saturation
    scale = 0.1
    v = torch.randn(batch_size, dim, device=device) * scale
    U = torch.randn(dim, rank, device=device) * scale
    W = torch.randn(dim, rank, device=device) * scale
    
    # Run CUDA kernel
    gamma_cuda = christoffel_fused(v, U, W)
    
    # Run PyTorch Fallback
    v_cpu = v.cpu()
    U_cpu = U.cpu()
    W_cpu = W.cpu()
    
    proj = torch.matmul(v_cpu, U_cpu)
    sq = proj * proj
    gamma_ref = torch.matmul(sq, W_cpu.t())
    
    print(f"Ref Gamma Range: [{gamma_ref.min():.4f}, {gamma_ref.max():.4f}]")
    gamma_ref_clamped = torch.clamp(gamma_ref, -5.0, 5.0)
    
    diff = (gamma_cuda.cpu() - gamma_ref_clamped).abs().max().item()
    print(f"Max Difference (Christoffel): {diff:.6f}")
    
    if diff < 1e-3:
        print(">> Christoffel Kernel: PASS")
    else:
        print(">> Christoffel Kernel: FAIL")
        print(f"CUDA sample: {gamma_cuda[0, :5]}")
        print(f"Ref sample:  {gamma_ref_clamped[0, :5]}")

    # --- Verify Leapfrog Fused ---
    print("\n--- Verifying Leapfrog Fused ---")
    dt = 0.01
    dt_scale = 1.0
    
    x = torch.randn(batch_size, dim, device=device) * scale
    v = torch.randn(batch_size, dim, device=device) * scale
    f = torch.randn(batch_size, dim, device=device) * scale
    
    # Re-use U, W from above (scaled)
    
    # Run CUDA kernel
    x_new_cuda, v_new_cuda = leapfrog_fused(x, v, f, U, W, dt, dt_scale)
    
    # Run PyTorch Ref
    effective_dt = dt * dt_scale
    
    # Half-step velocity
    # Redo calc to ensure we use exact same inputs
    v_cpu = v.cpu()
    f_cpu = f.cpu()
    x_cpu = x.cpu()
    
    gamma_v = torch.matmul((torch.matmul(v_cpu, U_cpu)**2), W_cpu.t())
    gamma_v = gamma_v.clamp(-5, 5)
    
    v_half_ref = v_cpu + 0.5 * effective_dt * (f_cpu - gamma_v)
    
    # Full-step position
    x_new_ref = x_cpu + effective_dt * v_half_ref
    
    # Half-step velocity again
    gamma_v_half = torch.matmul((torch.matmul(v_half_ref, U_cpu)**2), W_cpu.t())
    gamma_v_half = gamma_v_half.clamp(-5, 5)
    
    v_new_ref = v_half_ref + 0.5 * effective_dt * (f_cpu - gamma_v_half)
    
    diff_x = (x_new_cuda.cpu() - x_new_ref).abs().max().item()
    diff_v = (v_new_cuda.cpu() - v_new_ref).abs().max().item()
    
    print(f"Max Difference (Leapfrog X): {diff_x:.6f}")
    print(f"Max Difference (Leapfrog V): {diff_v:.6f}")
    
    if diff_x < 1e-3 and diff_v < 1e-3:
        print(">> Leapfrog Kernel: PASS")
    else:
        print(">> Leapfrog Kernel: FAIL")
        print(f"CUDA v_new sample: {v_new_cuda[0, :5]}")
        print(f"Ref v_new sample:  {v_new_ref[0, :5]}")

if __name__ == "__main__":
    verify_kernels()
