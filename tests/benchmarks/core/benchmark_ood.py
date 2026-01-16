
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.math_dataset import MathDataset
from src import GFN
try:
    from tests.benchmarks.baselines import MicroGPT
except ImportError:
    from baselines import MicroGPT

def evaluate_accuracy(model, digits, samples=100, device='cuda'):
    """Evaluates model accuracy on addition problems of specific digit length."""
    dataset = MathDataset(size=samples, max_digits=digits)
    model.eval()
    
    correct = 0
    total = 0
    
    print(f"  Testing {digits}-digit addition...", end="", flush=True)
    
    with torch.no_grad():
        for i in range(samples):
            # Get single item
            problem_str = dataset._generate_problem()
            parts = problem_str.split('=')
            prompt_str = parts[0] + '='
            target_str = parts[1]
            
            # Encode
            ids = [dataset.char_to_id[c] for c in prompt_str]
            input_seq = torch.tensor([ids]).to(device)
            
            # Simple greedy generation
            # For GFN/GPT, we generate until prediction length matches target length roughly
            # or hit EOS. For addition, result length is roughly digits + 1
            
            if isinstance(model, GFN):
                 logits, state = model(input_seq)
            else: # GPT
                 # GPT needs loop from scratch or KV cache (simplified loop here)
                 pass # Placeholder for GPT generation logic
            
            # ... (Generation Logic Simplified for brevity) ...
            # Implementation Note: Full generation loop is needed here.
            # Reuse run_demo logic from train.py but strictly for accuracy.
            
            # Using a simplified check for now: 
            # We assume the model relies on the state passed.
            
            generated = list(ids)
            state = None
            
            # Initial pass
            if isinstance(model, GFN):
                logits, state = model(input_seq)
                curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            else:
                # Naive GPT generation (slow but correct)
                # Recalculate full sequence each time
                logits = model(input_seq)
                curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

            generated.append(curr_token.item())
            
            # Generate rest
            max_gen = digits + 2 
            for _ in range(max_gen):
                if isinstance(model, GFN):
                    logits, state = model(curr_token, state=state)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                else:
                    inp = torch.tensor([generated]).to(device)
                    logits = model(inp)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                
                tok_id = next_token.item()
                if tok_id == dataset.char_to_id.get('<EOS>', -1) or tok_id == dataset.char_to_id.get('<PAD>', -1):
                    break
                generated.append(tok_id)
                curr_token = next_token.unsqueeze(0)
                
            pred_res = dataset.decode(generated).split('=')[-1]
            if pred_res == target_str:
                correct += 1
            total += 1
            
    acc = (correct / total) * 100
    print(f" Acc: {acc:.1f}%")
    return acc

def run_ood_suite(checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running OOD Generalization Benchmark...")
    
    # Load GFN Model
    # Assumes we use the Medium config
    # Ideally load config from checkpoint but hardcoding for demo
    gfn_model = GFN(vocab_size=20, dim=512, depth=12, rank=16).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading GFN checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
           gfn_model.load_state_dict(ckpt['model_state_dict'])
        except:
           print("Warning: Could not load weights (architecture mismatch?)")
    else:
        print("Warning: No checkpoint found, benchmarking initialized (random) model.")

    # We skip GPT training for now, just compare GFN across digits
    models = {"GFN-Medium": gfn_model}
    
    # Test on 2 (Train), 3 (OOD), 4 (OOD), 5 (Deep OOD)
    difficulties = [2, 3, 4, 5]
    results = []
    
    for name, model in models.items():
        for d in difficulties:
            acc = evaluate_accuracy(model, d, samples=50, device=device)
            results.append({
                "Model": name,
                "Digit Length": d,
                "Accuracy (%)": acc,
                "Type": "Train Distribution" if d <= 2 else "OOD"
            })
            
    # Visualize
    df = pd.DataFrame(results)
    os.makedirs("tests/professional/results", exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Digit Length", y="Accuracy (%)", hue="Model")
    plt.title("Systemic Generalization (OOD)", fontsize=14)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Training Boundary')
    plt.savefig("tests/professional/results/ood_generalization.png")
    plt.close()
    print("OOD Benchmark Complete.")

if __name__ == "__main__":
    # Expects checkpoint path as arg or defaults
    ckpt = "checkpoints/medium_fast/epoch_100.pt" 
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    run_ood_suite(ckpt)
