import torch
import matplotlib.pyplot as plt
from src.model import Manifold
from src.math_dataset import MathDataset

def visualize_gating(model_path, test_str="88+11="):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dummy dataset for vocab/params
    dataset = MathDataset(size=10, max_digits=2)
    vocab_size = dataset.vocab_size
    
    # Init and Load Model
    # Assuming standard architecture (dim=128, depth=4, etc.)
    # In a real scenario, we'd load these from a config file
    model = Manifold(vocab_size, dim=128, depth=4, rank=64).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except:
        print("Could not load weights, using random model for visualization.")
    
    model.eval()
    
    ids = [dataset.char_to_id[c] for c in test_str]
    input_ids = torch.tensor([ids]).to(device)
    
    # Hook to capture gating values
    gates_log = []
    
    def hook_fn(module, input, output):
        # output is the gate value [batch, 1] (or dt_scale)
        gates_log.append(output.detach().cpu().numpy())

    # Register hooks on all RiemannianGating modules OR TimeDilationHead
    hooks = []
    for name, module in model.named_modules():
        mod_type = str(type(module))
        if "RiemannianGating" in mod_type or "TimeDilationHead" in mod_type:
            hooks.append(module.register_forward_hook(hook_fn))
            
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    # Clean up hooks
    for h in hooks:
        h.remove()
        
    # Process gates_log
    # gates_log will have: [layer0_t0, layer1_t0, ..., layerN_t0, layer0_t1, ...]
    # Reshape: [seq_len, num_layers]
    num_layers = len([m for m in model.modules() if "GLayer" in str(type(m))])
    num_steps = len(ids)
    
    gates_array = torch.tensor(gates_log).view(num_steps, num_layers).numpy()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.imshow(gates_array.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Gate Opening (Flow Speed)')
    plt.title(f"Riemannian Gating Flow: '{test_str}'")
    plt.xlabel("Token Steps")
    plt.ylabel("Layer Depth")
    plt.xticks(range(num_steps), list(test_str))
    plt.tight_layout()
    
    save_path = "gating_visualization.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # You can change the path to your last saved epoch!
    visualize_gating("gfn_math.pth")
