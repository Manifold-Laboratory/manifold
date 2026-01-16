import torch
import sys
import os
from pathlib import Path
import random

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold
from tests.benchmarks.bench_utils import measure_peak_memory

def evaluate_accuracy(checkpoint_path, test_type="math_oracle"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Benchmarking Manifold Accuracy: {test_type}")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    token_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_token = {i: c for c, i in token_to_id.items()}
    
    model = Manifold(vocab_size=len(vocab), dim=512, depth=8, heads=8).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # 2. Test Cases (In-Distribution vs Out-of-Distribution/Zero-shot)
    test_suites = {
        "Base (2-digit)": [
            "12 + 15 = 27", "45 - 12 = 33", "20 * 3 = 60", "99 + 01 = 100"
        ],
        "Generalization (3-digit)": [
            "123 + 456 = 579", "999 - 111 = 888", "100 * 5 = 500"
        ],
        "Extreme (4-digit+)": [
            "1234 + 5678 = 6912", "9999 + 0001 = 10000"
        ]
    }
    
    overall_results = {}
    
    for suite_name, problems in test_suites.items():
        print(f"\nTesting {suite_name}...")
        correct = 0
        total = len(problems)
        
        for prob in problems:
            # We split prob into "input = " and "answer"
            # In a real setup, we'd attend up to '=' and predict.
            # Here we'll check the 'Energy' of the complete sequence vs perturbed ones.
            # The model should assign LOWEST energy to the CORRECT sequence.
            
            try:
                # VRAM Measurement
                ids = torch.tensor([token_to_id[c] for c in prob]).unsqueeze(0).to(device)
                peak_mem = 0.0
                def forward_pass():
                    return model(ids)
                
                # Measure memory if user requested benchmarks
                peak_mem = measure_peak_memory(model, forward_pass)

                logits = forward_pass()
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Check prediction for sequence (shifted)
                pred_ids = torch.argmax(logits[:, :-1, :], dim=-1)
                target_ids = ids[:, 1:]
                
                match = (pred_ids == target_ids).all().item()
                if match:
                    correct += 1
                
                print(f"  [{'✓' if match else '✗'}] {prob} | VRAM: {peak_mem:.1f} MB")
            except Exception as e:
                print(f"  [Error] {prob}: {e}")
                
        acc = (correct / total) * 100
        overall_results[suite_name] = acc
        print(f">> Accuracy: {acc:.2f}%")

    print("\n" + "=" * 60)
    print("🏆 ACCURACY BENCHMARK SUMMARY")
    print("=" * 60)
    for suite, acc in overall_results.items():
        print(f"{suite:<25} | {acc:>6.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    ckpt = "checkpoints\medium_fast\v0.5\epoch_4.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    evaluate_accuracy(ckpt)
