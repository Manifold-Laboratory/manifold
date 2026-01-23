import torch
from gfn.model import Manifold

def diagnose():
    print("Diagnosing Manifold Dynamics...")
    # Initialize model with v2.6.2 defaults (Yoshida)
    model = Manifold(vocab_size=1000, dim=256, depth=8, integrator_type='yoshida')
    model.eval()

    # Create dummy input
    curr_token = torch.randint(0, 1000, (1, 1))
    
    # Run forward pass manually to inspect intermediate states
    print("\n--- Layer-by-Layer Dynamics ---")
    x = model.x0.expand(1, -1)
    v = model.v0.expand(1, -1)
    
    # Force for first token
    force = model.embedding(curr_token).squeeze(1)
    
    print(f"Init: |x|={x.norm().item():.4f}, |v|={v.norm().item():.4f}, |F|={force.norm().item():.4f}")
    
    for i, layer in enumerate(model.layers):
        x, v, ctx, christ = layer(x, v, force=force)
        
        v_mag = v.norm().item()
        x_mag = x.norm().item()
        print(f"Layer {i+1}: |x|={x_mag:.4f}, |v|={v_mag:.4f}")
        
        # Check for explosion
        if v_mag > 10.0:
            print(f"!!! EXPLOSION DETECTED AT LAYER {i+1} !!!")
        if v_mag < 0.001:
            print(f"!!! VANISHING DYNAMICS AT LAYER {i+1} !!!")

diagnose()
