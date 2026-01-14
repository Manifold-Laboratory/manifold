
import unittest
import torch
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold

class TestRecursiveGeodesics(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'active_inference': {
                'enabled': True,
                'recursive_geodesics': {
                    'enabled': True
                },
                # Disable other active features for isolation
                'reactive_curvature': {'enabled': False},
                'singularities': {'enabled': False},
                'dynamic_time': {'enabled': False}
            }
        }
        
    def test_recursive_context_propagation(self):
        """Test if context propagates and influences output."""
        dim = 32
        model = Manifold(
            vocab_size=10, 
            dim=dim, 
            depth=3, # Need >1 layer to test recursion
            physics_config=self.config
        )
        
        # Check if internal flags are set
        self.assertTrue(model.layers[0].use_recursive)
        
        # Forward pass
        input_ids = torch.randint(0, 10, (1, 5))
        
        # We want to verify that context is actually being used.
        # It's hard to verify "usage" without hooks or gradient checks.
        # But we can verify it runs without crashing (dimension mismatch).
        logits, state = model(input_ids)
        self.assertEqual(logits.shape, (1, 5, 10))
        
        # Verify gradients flow through context_proj?
        # Only if we do backward.
        loss = logits.sum()
        loss.backward()
        
        # Check if context projection weights got gradients
        # Layer 1's context_proj should receive gradients from Layer 2
        # Layer 0's context_proj receives gradients from Layer 1? 
        # Layer 0 receives context=None, so its context_proj input is None?
        # Wait, forward loop:
        # ctx=None -> Layer 0 -> ctx0 -> Layer 1 -> ctx1 -> Layer 2
        # Layer 0: context=None. force = force + proj(None)? No, check logic.
        # "if context is not None:"
        # So Layer 0 context_proj is NOT used. Grads should be None or Zero.
        
        # Layer 1: receives ctx0. ctx0 depends on Layer 0's gates.
        # So Layer 1's context_proj should have gradients.
        
        # Let's check Layer 1
        grad_L1 = model.layers[1].context_proj.weight.grad
        self.assertIsNotNone(grad_L1)
        self.assertNotEqual(grad_L1.abs().sum().item(), 0.0)
        
        # Layer 0 context_proj should be unused (None grad or zero)
        grad_L0 = model.layers[0].context_proj.weight.grad
        # It might be None because it never participated in the graph
        self.assertTrue(grad_L0 is None or grad_L0.abs().sum().item() == 0.0)

if __name__ == '__main__':
    unittest.main()
