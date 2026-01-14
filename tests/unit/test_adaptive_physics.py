import torch
import torch.nn as nn
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.geometry import LowRankChristoffel, DormandPrinceIntegrator

class TestAdaptivePhysics(unittest.TestCase):
    def test_dormand_prince_integration(self):
        """Verify Dormand-Prince integrator executes and produces valid output"""
        dim = 16
        net = LowRankChristoffel(dim, rank=4)
        integrator = DormandPrinceIntegrator(net, dt=0.1)
        
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        
        # Run one step
        x_new, v_new = integrator(x, v)
        
        # Check shapes
        self.assertEqual(x_new.shape, x.shape)
        self.assertEqual(v_new.shape, v.shape)
        
        # Check values are finite (not NaN/Inf)
        self.assertTrue(torch.isfinite(x_new).all())
        self.assertTrue(torch.isfinite(v_new).all())
        
        # Check that it moved positions
        diff = (x_new - x).abs().max().item()
        self.assertGreater(diff, 1e-6, "Integrator should update position")

    def test_rk45_accuracy(self):
        """Verify RK45 produces similar results to a simple Euler step for small dt"""
        dim = 16
        net = LowRankChristoffel(dim, rank=4)
        integrator = DormandPrinceIntegrator(net, dt=0.001) # Very small step
        
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        
        # RK45 Step
        x_rk, v_rk = integrator(x, v)
        
        # Simple Euler Step
        # a = -Gamma(v)
        # v_ec = v + a*dt
        # x_ec = x + v*dt
        a = -net(v, x)
        v_euler = v + a * 0.001
        x_euler = x + v * 0.001
        
        # Should be relatively close for tiny dt
        # RK45 is much more accurate, but for tiny dt both converge
        torch.testing.assert_close(x_rk, x_euler, rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
    unittest.main()
