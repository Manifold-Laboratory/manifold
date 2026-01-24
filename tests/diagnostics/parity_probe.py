import torch
import torch.nn.functional as F
import numpy as np
from gfn.model import Manifold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def probe_latent_clusters():
    """Visualize and measure separability of final states x_L for parity."""
    print("====================================================")
    print("   GFN LATENT SEPARABILITY PROBE (Phase 12)        ")
    print("====================================================")
    
    L = 10
    dim = 128
    # Using the current standard config
    model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32,
                     physics_config={'readout': {'type': 'implicit'}})
    model.eval()
    
    # 1. Generate Batch of sequences
    N = 256
    inputs = torch.randint(0, 2, (N, L))
    targets = (inputs.sum(dim=-1) % 2).long()
    
    with torch.no_grad():
        # Get latent states
        # We need the FINAL state x_L
        # model returns (logits, (state_x, state_v), christoffels)
        # where state_x is the state AFTER the last step
        _, (state_x, _), _ = model(inputs)
        
    x_latent = state_x.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # 2. Measure Euclidean distance between centers
    x0 = x_latent[targets_np == 0]
    x1 = x_latent[targets_np == 1]
    
    center0 = x0.mean(axis=0)
    center1 = x1.mean(axis=0)
    
    inter_dist = np.linalg.norm(center0 - center1)
    intra_std0 = x0.std(axis=0).mean()
    intra_std1 = x1.std(axis=0).mean()
    
    print(f"\n--- CLUSTER METRICS (L={L}) ---")
    print(f"Inter-Cluster Distance (Centers): {inter_dist:.4f}")
    print(f"Intra-Cluster Spread (Std):    {(intra_std0 + intra_std1)/2:.4f}")
    
    sep_ratio = inter_dist / (intra_std0 + intra_std1 + 1e-9)
    status = "\033[92m[GOOD]\033[0m" if sep_ratio > 1.2 else "\033[91m[COLLAPSED]\033[0m"
    print(f"Separability Ratio:            {sep_ratio:.4f} {status}")
    
    if sep_ratio < 0.5:
        print("\n\033[91mWARNING:\033[0m Manifold states for Parity 0/1 are nearly identical.")
        print("The manifold is not 'steering' the particle based on inputs.")

    # 3. PCA Projection for Visual confirmation (if terminal supports it? No, just print summary)
    try:
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x_latent)
        expl_var = pca.explained_variance_ratio_.sum()
        print(f"PCA Variance Explained (2D): {expl_var:.2%}")
    except:
        pass

def probe_force_signal_ratio():
    """Measure the ratio of Christoffel force vs Input force."""
    print("\n--- SIGNAL RATIO TEST ---")
    dim = 128
    model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32)
    model.eval()
    
    # 1. Measure Input Force Magnitude
    # F = embedding(1)
    force_1 = model.embedding(torch.tensor([[1]])) # [1, 1, 128]
    f_norm = force_1.norm().item()
    
    # 2. Measure Christoffel Force Magnitude at unit velocity
    v_unit = torch.randn(1, dim)
    v_unit = v_unit / v_unit.norm()
    
    with torch.no_grad():
        gamma = model.layers[0].christoffels[0](v_unit)
    g_norm = gamma.norm().item()
    
    print(f"Token Impulse Norm (|F|):    {f_norm:.4f}")
    print(f"Manifold Resitance Norm (|Γ|): {g_norm:.4f}")
    
    ratio = f_norm / (g_norm + 1e-9)
    print(f"Force/Curvature Ratio:        {ratio:.2f}x")
    
    if ratio > 10.0:
        print("\033[93m[IMBALANCE]\033[0m Token force dominates geometry. Manifold is too 'soft'.")
    elif ratio < 0.1:
        print("\033[93m[IMBALANCE]\033[0m Geometry dominates tokens. Manifold is too 'stiff'.")
    else:
        print("\033[92m[BALANCED]\033[0m Dynamics are in the steerable regime.")

if __name__ == "__main__":
    try:
        probe_force_signal_ratio()
        probe_latent_clusters()
    except Exception as e:
        print(f"Probe failed: {e}")
        import traceback
        traceback.print_exc()
    print("\n====================================================")
