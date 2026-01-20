"""
Example: Load and use a saved Manifold checkpoint
"""
import torch
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold

def load_manifold_checkpoint(checkpoint_path):
    """
    Load a saved Manifold model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        
    Returns:
        model: Loaded Manifold model
        checkpoint: Full checkpoint dict with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config
    config = checkpoint['model_config']
    
    # Recreate model
    model = Manifold(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        integrator_type=config['integrator_type'],
        physics_config=config['physics_config']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded Manifold checkpoint")
    print(f"   Task: {checkpoint.get('task', 'Unknown')}")
    print(f"   Final Loss: {checkpoint.get('final_loss', 'N/A'):.4f}")
    
    return model, checkpoint


if __name__ == "__main__":
    # Example usage
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "manifold_parity_superiority.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Run the superiority benchmark first to train and save the model.")
        sys.exit(1)
    
    # Load model
    model, checkpoint = load_manifold_checkpoint(checkpoint_path)
    model.eval()
    
    # Test inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate a test sequence
    test_input = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 0]], device=device)
    
    with torch.no_grad():
        logits, (x_final, v_final), _ = model(test_input)
    
    # Decode predictions (binary threshold)
    preds = (logits[:, :, 0] > 0.0).long()
    
    print(f"\n📊 Test Inference:")
    print(f"   Input:  {test_input[0].tolist()}")
    print(f"   Output: {preds[0].tolist()}")
    
    # Expected output for Parity (cumulative XOR)
    expected = torch.cumsum(test_input, dim=1) % 2
    print(f"   Expected: {expected[0].tolist()}")
    print(f"   Accuracy: {(preds == expected).float().mean().item() * 100:.1f}%")
