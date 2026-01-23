
import torch
import torch.nn as nn
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from tests.benchmarks.viz.vis_gfn_superiority import ParityTask, train_step_manifold

def debug_backward():
    print("[-] Initializing Debug Environment...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Model with same config as benchmark
    dim = 128
    model = Manifold(
        vocab_size=2, dim=dim, depth=6, heads=4,
        use_scan=False,
        integrator_type='yoshida',
        physics_config={
            'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
            'readout': {'type': 'implicit', 'coord_dim': 16},
            'active_inference': {'enabled': True, 'plasticity': 0.1},
            'fractal': {'enabled': True},
            'singularities': {'enabled': True, 'strength': 5.0},
            'stability': {'base_dt': 0.05}
        }
    ).to(device)
    
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3, max_norm=10.0)
    
    # Generate Dummy Data
    task = ParityTask(length=20)
    x, y = task.generate_batch(32, device=device)
    
    print("[-] Running Forward/Backward Step...")
    # Run Step
    loss, l_mse, l_phy, acc, grad_norm = train_step_manifold(model, optimizer, None, x, y, device)
    
    print(f"[-] Step Complete. Loss: {loss:.4f} | GradNorm: {grad_norm:.4f}")
    
    print("\n[-] Gradient Audit:")
    print(f"{'Parameter Name':<50} | {'Grad Norm':<15} | {'Status'}")
    print("-" * 80)
    
    zero_grads = []
    none_grads = []
    
    for name, p in model.named_parameters():
        if p.grad is None:
            status = "NONE (Disconnected)"
            norm_val = 0.0
            none_grads.append(name)
        else:
            norm_val = p.grad.data.norm().item()
            if norm_val == 0.0:
                status = "ZERO (Vanished)"
                zero_grads.append(name)
            else:
                status = "OK"
        
        print(f"{name:<50} | {norm_val:<15.6f} | {status}")

    print("\n[-] Summary:")
    print(f"Total Parameters: {len(list(model.named_parameters()))}")
    print(f"Disconnected (None): {len(none_grads)}")
    print(f"Vanished (Zero): {len(zero_grads)}")
    
    if len(none_grads) > 0:
        print("\n[!] Disconnected Params:")
        for n in none_grads: print(f"  - {n}")
        
    if len(zero_grads) > 0:
        print("\n[!] Zero Gradient Params:")
        for n in zero_grads: print(f"  - {n}")

if __name__ == "__main__":
    debug_backward()
