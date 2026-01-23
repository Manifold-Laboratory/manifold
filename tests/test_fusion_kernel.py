"""
Test if recurrent_manifold_fused kernel works at all
"""
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gfn_cuda
    print("✅ gfn_cuda imported successfully")
    print(f"Has recurrent_manifold_fused: {hasattr(gfn_cuda, 'recurrent_manifold_fused')}")
    
    # Try calling the kernel directly
    device = torch.device('cuda')
    batch, dim, seq_len, rank, num_layers, num_heads = 2, 128, 10, 32, 6, 4
    dim_per_head = dim // num_heads
    
    x = torch.randn(batch, dim, device=device)
    v = torch.randn(batch, dim, device=device)
    forces = torch.randn(batch, seq_len, dim, device=device)
    
    # Stack for multi-head: [num_layers * num_heads, dim_per_head, rank]
    U_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
    W_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
    
    print(f"\nCalling kernel with:")
    print(f"  x: {x.shape}, v: {v.shape}")
    print(f"  forces: {forces.shape}")
    print(f"  U_stack: {U_stack.shape}, W_stack: {W_stack.shape}")
    print(f"  num_heads: {num_heads}")
    
    try:
        result = gfn_cuda.recurrent_manifold_fused(
            x, v, forces, U_stack, W_stack,
            0.3, 1.0, num_heads
        )
        print(f"\n✅ Kernel executed successfully!")
        print(f"  Result: {len(result)} tensors")
        print(f"  Shapes: {[r.shape for r in result]}")
    except Exception as e:
        print(f"\n❌ Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ Cannot import gfn_cuda: {e}")
