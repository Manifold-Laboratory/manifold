import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import yaml
import re
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import Manifold

class SimpleVocab:
    def __init__(self, tokens, min_freq=3):
        counter = Counter(tokens)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.stoi = {'<unk>': 0, '<pad>': 1, '<eos>': 2}
        self.itos = ['<unk>', '<pad>', '<eos>']
        
        idx = 3
        for token, freq in sorted_by_freq_tuples:
            if freq >= min_freq:
                self.stoi[token] = idx
                self.itos.append(token)
                idx += 1
        
    def __len__(self):
        return len(self.stoi)
        
    def __call__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])
        
    def decode(self, idx):
        if idx < len(self.itos):
            return self.itos[idx]
        return '<unk>'

def load_vocab():
    """Rebuild vocab from training data"""
    from demos.wikitext.train_wikitext import WikiText2Custom
    
    train_loader = WikiText2Custom(split='train')
    tokens = train_loader.read_tokens()
    vocab = SimpleVocab(tokens)
    return vocab

def generate_text(model, vocab, prompt, max_len=100, temperature=0.8, top_p=0.9, device='cuda'):
    model.eval()
    
    # Tokenize prompt
    tokens = re.findall(r'\S+', prompt.lower())
    input_ids = [vocab(t) for t in tokens]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_len):
            # Forward pass
            logits, _, _ = model(generated)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Nucleus sampling (top-p)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop at EOS
            if next_token.item() == vocab.stoi.get('<eos>', -1):
                break
    
    # Decode
    output_tokens = generated[0].cpu().tolist()
    output_text = ' '.join([vocab.decode(idx) for idx in output_tokens])
    return output_text

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load config
    config_path = 'configs/demos/wikitext.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load vocab
    print("Loading vocabulary...")
    vocab = load_vocab()
    config['model']['vocab_size'] = len(vocab)
    print(f"Vocab size: {len(vocab)}\n")
    
    # Initialize model
    print("Loading model...")
    model = Manifold(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        integrator_type=config['physics']['solver'],
        use_scan=config['model']['use_scan'],
        physics_config=config['physics']
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = f"{config['training']['save_dir']}/best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}\n")
    
    # Interactive generation
    print("=" * 80)
    print("Manifold Text Generation Demo")
    print("=" * 80)
    
    prompts = [
        "the united states",
        "in the year",
        "the president of",
        "artificial intelligence",
        "the theory of"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)
        generated = generate_text(model, vocab, prompt, max_len=50, temperature=0.8, top_p=0.9, device=device)
        print(generated)
        print()

if __name__ == '__main__':
    main()
