
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force CPU for stable visualization checks
device = torch.device('cpu')

from gfn.geometry import LowRankChristoffel

def benchmark_forget_gate():
    print("🔬 Benchmarking Dynamic Forget Gate (Friction)...")
    
    dim = 64
    batch_size = 1
    steps = 100
    
    # 1. Initialize Geometry
    # We expect friction to start low (bias=-3.0) and be learnable
    geo = LowRankChristoffel(dim, rank=16).to(device)
    
    # Manually tweak weights to simulate a "Trained" state
    # where high-norm inputs (surprise) trigger the gate
    with torch.no_grad():
        # Make the gate sensitive to input magnitude
        nn.init.eye_(geo.forget_gate.weight) 
        # Bias set to threshold activation around norm ~ 3.0
        geo.forget_gate.bias.fill_(-3.0) 
        
    # 2. Generate Synthetic "Context Switch" Data
    # Step 0-40: Low energy (Stable Topic)
    # Step 40-50: High energy spike (Context Switch)
    # Step 50-100: Low energy (New Topic)
    
    inputs = torch.zeros(steps, batch_size, dim).to(device)
    
    # Stable: Small random noise
    inputs[:40] = torch.randn(40, batch_size, dim) * 0.1
    inputs[50:] = torch.randn(50, batch_size, dim) * 0.1
    
    # Shock: Large update
    inputs[40:50] = torch.randn(10, batch_size, dim) * 5.0
    
    velocity = torch.ones(batch_size, dim).to(device) # Constant velocity to test drag
    
    friction_coefs = []
    total_resistance = []
    input_norms = []
    
    print("Running Simulation...")
    with torch.no_grad():
        for t in range(steps):
            x_t = inputs[t]
            
            # Forward pass to get generalized force (Gamma + Friction)
            # Output F_resist = Gamma(v,v) + Friction(x)*v
            resistance = geo(velocity, x_t)
            
            # Inspect internal Forget Gate
            # friction = sigmoid(Gate(x))
            gate_logits = geo.forget_gate(x_t)
            friction = torch.sigmoid(gate_logits).mean().item()
            
            friction_coefs.append(friction)
            total_resistance.append(resistance.norm().item())
            input_norms.append(x_t.norm().item())
            
    # 3. Visualization
    data = pd.DataFrame({
        "Step": range(steps),
        "Input Magnitude (Context Change)": input_norms,
        "Friction Coefficient (Forget Gate)": friction_coefs,
        "Total Resistance Force": total_resistance
    })
    
    plt.figure(figsize=(12, 6))
    
    # Left: Input Energy vs Friction
    plt.subplot(1, 2, 1)
    sns.lineplot(data=data, x="Step", y="Input Magnitude (Context Change)", label="Input Energy", color="blue", alpha=0.5)
    ax2 = plt.gca().twinx()
    sns.lineplot(data=data, x="Step", y="Friction Coefficient (Forget Gate)", label="Friction (Forget Gate)", color="red", ax=ax2, linewidth=2.5)
    
    plt.title("Dynamic Damping Response")
    plt.xlabel("Time Step")
    ax2.set_ylabel("Friction Coeff (0=Memory, 1=Forget)")
    ax2.set_ylim(-0.1, 1.1)
    
    # Right: Correlation
    plt.subplot(1, 2, 2)
    plt.scatter(data["Input Magnitude (Context Change)"], data["Friction Coefficient (Forget Gate)"], c=data["Step"], cmap="viridis", alpha=0.7)
    plt.title("Activation Function: Energy -> Forgetting")
    plt.xlabel("Input Magnitude")
    plt.ylabel("Friction (Damping)")
    plt.grid(True, alpha=0.3)
    
    output_dir = PROJECT_ROOT / "tests/benchmarks/results/stability"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dynamic_friction_test.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Benchmark Complete. Plot saved to: {out_path}")

if __name__ == "__main__":
    benchmark_forget_gate()
