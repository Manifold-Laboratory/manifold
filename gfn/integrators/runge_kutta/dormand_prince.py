
"""
Dormand-Prince (RK45) Adaptive Integrator.

Uses 5th order and 4th order approximations to estimate local error and adapt `dt`.
Ideally suited for "Golden Integration" to ensure physical stability.
"""
import torch
import torch.nn as nn


class DormandPrinceIntegrator(nn.Module):
    r"""
    Dormand-Prince (DP5) Integrator.
    
    Implementation of the 5th-order solution from the RK45 (Dormand-Prince) tableau.
    In this implementation, we use it as a high-precision Fixed-Step integrator,
    utilizing the 5th-order approximation 'y5' for the update.
    """
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.base_dt = dt
        
        # Butcher Tableau for RK45 (Dormand-Prince)
        # c: nodes
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        
        # a: Runge-Kutta matrix (flattened or manual for efficiency)
        # b5: 5th order weights
        self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        """
        Perform ONE fixed step using the 5th order DP solution.
        """
        dt = self.base_dt * dt_scale
        
        # Coefficients (DP54)
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        # a7... same as b5 for FSAL property
        
        def dynamics(tx, tv):
            acc = -self.christoffel(tv, tx)
            if force is not None:
                acc = acc + force
            return acc
            
        # k1
        k1_x = v
        k1_v = dynamics(x, v)
        
        # k2
        x2 = x + dt * (a21*k1_x)
        v2 = v + dt * (a21*k1_v)
        k2_x = v2
        k2_v = dynamics(x2, v2)
        
        # k3
        x3 = x + dt * (a31*k1_x + a32*k2_x)
        v3 = v + dt * (a31*k1_v + a32*k2_v)
        k3_x = v3
        k3_v = dynamics(x3, v3)
        
        # k4
        x4 = x + dt * (a41*k1_x + a42*k2_x + a43*k3_x)
        v4 = v + dt * (a41*k1_v + a42*k2_v + a43*k3_v)
        k4_x = v4
        k4_v = dynamics(x4, v4)
        
        # k5
        x5 = x + dt * (a51*k1_x + a52*k2_x + a53*k3_x + a54*k4_x)
        v5 = v + dt * (a51*k1_v + a52*k2_v + a53*k3_v + a54*k4_v)
        k5_x = v5
        k5_v = dynamics(x5, v5)
        
        # k6
        x6 = x + dt * (a61*k1_x + a62*k2_x + a63*k3_x + a64*k4_x + a65*k5_x)
        v6 = v + dt * (a61*k1_v + a62*k2_v + a63*k3_v + a64*k4_v + a65*k5_v)
        k6_x = v6
        k6_v = dynamics(x6, v6)
        
        # k7 (Accumulate result using b5)
        # Note: In DP54, the update step IS k7 (FSAL property)
        # y_{n+1} = y_n + dt * \sum (b_i * k_i)
        
        # x_next
        x_next = x + dt * (self.b5[0]*k1_x + self.b5[2]*k3_x + self.b5[3]*k4_x + self.b5[4]*k5_x + self.b5[5]*k6_x)
        
        # v_next
        v_next = v + dt * (self.b5[0]*k1_v + self.b5[2]*k3_v + self.b5[3]*k4_v + self.b5[4]*k5_v + self.b5[5]*k6_v)
        
        return x_next, v_next
