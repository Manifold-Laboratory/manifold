import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.integrators.runge_kutta.rk4 import RK4Integrator
from gfn.integrators.runge_kutta.heun import HeunIntegrator
from gfn.geometry import LowRankChristoffel

def test_smoothness_impact():
    print("🔬 Debugging Integrator Stability: The Smoothness Bottleneck")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dim = 64
    batch_size = 1
    dt = 0.01
    steps = 100
    
    # 1. Setup two versions: Static (Smooth) and Active (Non-Smooth)
    from gfn.geometry import ReactiveChristoffel
    
    physics_config_active = {
        'active_inference': {
            'enabled': True,
            'reactive_curvature': {'enabled': True, 'plasticity': 1.0},
            'singularities': {'enabled': True, 'strength': 10.0, 'threshold': 0.1} # High sensitivity
        }
    }
    
    geo_static = LowRankChristoffel(dim).to(device)
    geo_active = ReactiveChristoffel(dim, physics_config=physics_config_active).to(device)
    
    # Init with standard values
    with torch.no_grad():
        nn.init.normal_(geo_static.U, std=1.0) # Medium curvature
        nn.init.normal_(geo_static.W, std=1.0)
        geo_active.U.data.copy_(geo_static.U.data)
        geo_active.W.data.copy_(geo_static.W.data)
        # Trigger potential for singularities
        nn.init.normal_(geo_active.V.weight, std=1.0) 

    results = []

    for name, geo in [("Static (Smooth)", geo_static), ("Active (Fractal)", geo_active)]:
        for int_name, IntegratorClass in [("Heun", HeunIntegrator), ("RK4", RK4Integrator)]:
            integrator = IntegratorClass(geo, dt=dt).to(device)
            
            x = torch.zeros(batch_size, dim).to(device)
            v = torch.randn(batch_size, dim).to(device)
            v = v / (v.norm() + 1e-8) # Normal energy
            
            v_init_norm = v.norm().item()
            history = [v_init_norm]
            energy_gain_steps = 0
            
            # Initial acceleration check
            with torch.no_grad():
                acc_init = -geo(v, x)
                work_init = torch.sum(v * acc_init).item()
                print(f"   [*] {name} + {int_name} -> Initial v·a: {work_init:.8f} ({'Accelerating' if work_init > 0 else 'Decelerating'})")

            failed = False
            for s in range(steps):
                try:
                    # Capture acceleration sign before step
                    acc = -geo(v, x)
                    if torch.sum(v * acc) > 0:
                        energy_gain_steps += 1
                        
                    x, v = integrator(x, v)
                    norm = v.norm().item()
                    history.append(norm)
                    if norm > 1e10 or torch.isnan(v).any():
                        print(f"   [!] exploded at step {s}")
                        failed = True
                        break
                except Exception as e:
                    print(f"   [!] failed: {e}")
                    failed = True
                    break
            
            final_drift = (history[-1] - v_init_norm) / v_init_norm * 100 if not failed else float('inf')
            results.append({
                "Geometry": name,
                "Integrator": int_name,
                "Drift (%)": final_drift,
                "Gain Steps": f"{energy_gain_steps}/{steps}",
                "Status": "Exploded" if failed else "Stable"
            })

    print("\n📈 Results:")
    print(f"{'Geometry':<15} | {'Integrator':<10} | {'Drift (%)':<15} | {'Status'}")
    print("-" * 60)
    for r in results:
        drift_str = f"{r['Drift (%)']:.2f}" if r['Drift (%)'] != float('inf') else "inf"
        print(f"{r['Geometry']:<15} | {r['Integrator']:<10} | {drift_str:<15} | {r['Status']}")
    
    # Save a verification plot
    plt.figure(figsize=(10, 6))
    plt.title("Drift Comparison: C2 vs C-inf Geometry")
    plt.ylabel("Velocity Norm")
    plt.yscale("log")
    # ... plotting logic if needed ...

if __name__ == "__main__":
    test_smoothness_impact()
