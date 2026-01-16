
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import LowRankChristoffel, HeunIntegrator, RK4Integrator, LeapfrogIntegrator, SymplecticIntegrator

def run_strict_benchmark():
    print("[*] Running STRICT Symplectic Benchmark (Bypassing LayerNorm)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dim = 128
    rank = 32
    dt = 0.1
    steps = 200
    
    # Shared Geometry (Learned Connection)
    # We use the same random geometry for all to be fair
    torch.manual_seed(42)
    christoffel = LowRankChristoffel(dim, rank).to(device)
    
    # Integrators to test
    integrators = {
        'heun': HeunIntegrator(christoffel, dt),
        'rk4': RK4Integrator(christoffel, dt),
        'leapfrog': LeapfrogIntegrator(christoffel, dt),
        'symplectic': SymplecticIntegrator(christoffel, dt)
    }
    
    results = {}
    
    for name, integ in integrators.items():
        # Reset state
        torch.manual_seed(123)
        x = torch.randn(1, dim).to(device) # Position
        v = torch.randn(1, dim).to(device) # Velocity
        
        # Initial Energy (Approximate as Kinetic)
        E0 = 0.5 * (v.norm().item() ** 2)
        v0_norm = v.norm().item()
        
        energies = []
        drifts = []
        
        curr_x, curr_v = x, v
        
        with torch.no_grad():
            for t in range(steps):
                curr_x, curr_v = integ(curr_x, curr_v)
                
                # Check "Energy"
                # Since we don't have a metric tensor G, we track Euclidean Kinetic Energy
                # Note: In a general affine connection, this is NOT strictly conserved,
                # but Symplectic methods should show BOUNDED oscillation vs Drift.
                
                E_t = 0.5 * (curr_v.norm().item() ** 2)
                energies.append(E_t)
                
                # Relative error
                drift = (E_t - E0) / E0
                drifts.append(abs(drift))
        
        results[name] = {
            'final_drift_pct': drifts[-1] * 100,
            'max_drift_pct': max(drifts) * 100,
            'energies': energies
        }
        print(f"   {name.upper()}: Final Drift {drifts[-1]*100:.4f}% | Max Drift {max(drifts)*100:.4f}%")

    # === Plot ===
    print("\n[*] Generating Strict Plot...")
    res_dir = PROJECT_ROOT / "tests/benchmarks/results/integrators"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res['energies'], label=f"{name} (Drift: {res['final_drift_pct']:.1f}%)")
        
    plt.title(f"Strict Integrator Stability (Steps={steps}, dt={dt})")
    plt.xlabel("Step")
    plt.ylabel("Kinetic Energy (0.5 * v^2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(res_dir / "integrator_strict_comparison.png")
    print(f"[*] Saved to {res_dir / 'integrator_strict_comparison.png'}")

if __name__ == "__main__":
    run_strict_benchmark()
