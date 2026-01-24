import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gfn
from gfn.model import Manifold
import inspect

print(f"[*] GFN Path: {gfn.__file__}")
print(f"[*] Manifold File: {inspect.getfile(Manifold)}")

source = inspect.getsource(Manifold.forward)
print("[*] Manifold.forward First 10 Lines:")
print("\n".join(source.splitlines()[:10]))

print("\n[*] CALLING Manifold.forward NOW...")
import torch
model = Manifold(vocab_size=2, dim=64, depth=1)
model.to('cuda')
inputs = torch.zeros(1, 5).long().to('cuda')
logits, _, _ = model(inputs)
print("[*] CALL COMPLETE.")

