import torch
import torch.nn as nn
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.layers import MLayer

class TestGoldenIntegration(unittest.TestCase):
    def test_mlayer_rk45_init(self):
        """Verify MLayer initializes with rk45 integrator"""
        dim = 16
        heads = 4
        layer = MLayer(dim, heads=heads, integrator_type='rk45')
        
        # Check type of first integrator
        integ_name = layer.integrators[0].__class__.__name__
        self.assertEqual(integ_name, 'DormandPrinceIntegrator')

    def test_mlayer_rk45_forward(self):
        """Verify MLayer runs forward pass using RK45"""
        dim = 16
        heads = 4
        layer = MLayer(dim, heads=heads, integrator_type='rk45')
        
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        force = torch.randn(2, dim)
        
        # Run forward pass
        x_out, v_out = layer(x, v, force=force)
        
        # Check shapes
        self.assertEqual(x_out.shape, x.shape)
        self.assertEqual(v_out.shape, v.shape)
        
        # Check gradients flow
        loss = x_out.sum()
        loss.backward()
        
        # Check parameters have grad
        self.assertIsNotNone(layer.out_proj_x.weight.grad)

if __name__ == '__main__':
    unittest.main()
