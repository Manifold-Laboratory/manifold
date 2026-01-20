
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MicroGPT(nn.Module):
    """
    A lightweight GPT implementation (Decoder-only Transformer) 
    designed to match the parameter count of GFN-Medium for fair comparison.
    """
    def __init__(self, vocab_size, dim, depth, heads, max_len=10000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        self.drop = nn.Dropout(0.1)
        
        # Transformer Block
        # norm_first=True (Pre-LN) is standard for GPT-2/3 stability
        layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=4*dim, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True, 
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        b, t = idx.size()
        if t > self.pos_emb.size(1):
            raise ValueError(f"Sequence length {t} exceeds max length {self.pos_emb.size(1)}")
            
        x = self.token_emb(idx) + self.pos_emb[:, :t, :]
        x = self.drop(x)
        
        # Causal mask (ensure autoregressive property)
        # 0 on diagonal and below, -inf above
        mask = torch.triu(torch.ones(t, t, device=idx.device) * float('-inf'), diagonal=1)
        
        # is_causal=True is optimized in PyTorch 2.0+
        x = self.blocks(x, mask=mask, is_causal=True)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class MicroMamba(nn.Module):
    """
    Minimal Selective SSM (Mamba-like) implementation in pure PyTorch.
    Optimized for O(1) inference demonstration.
    
    Structure:
    x -> proj(2D) -> conv1d -> silu -> ssm -> ... -> proj(D)
    """
    def __init__(self, vocab_size, dim, depth, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.expand = expand
        self.d_inner = dim * expand
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            MambaBlock(dim, d_state, d_conv, expand) 
            for _ in range(depth)
        ])
        
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids, state=None):
        """
        Args:
            input_ids: [B, L]
            state: List of layer states. Each layer state is (h, conv_state).
                   h: [B, D_inner, D_state]
                   conv_state: [B, D_inner, D_conv]
                   
        Returns:
            logits: [B, L, V]
            new_state: Updated state list
        """
        x = self.embedding(input_ids)
        
        next_state = []
        current_state_idx = 0
        
        for layer in self.layers:
            layer_state = state[current_state_idx] if state is not None else None
            x, new_layer_state = layer(x, layer_state)
            next_state.append(new_layer_state)
            current_state_idx += 1
            
        x = self.norm_f(x)
        logits = self.head(x)
        
        return logits, next_state

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()
        
    def forward(self, x, state=None):
        # x: [B, L, D]
        batch, seq_len, _ = x.shape
        
        # 1. Project
        xz = self.in_proj(x) # [B, L, 2*D_in]
        x_in, z = xz.chunk(2, dim=-1) # [B, L, D_in]
        
        # 2. Conv1D (Causal)
        if state is None:
             # Full sequence mode
             x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
             x_conv = x_conv[:, :seq_len, :] # Crop padding
             new_conv_state = x_in[:, -self.conv1d.kernel_size[0]+1:, :] # Last K-1 tokens
        else:
             # Step mode (O(1))
             # Shift conv buffer
             prev_conv, prev_ssm = state
             # prev_conv: [B, K-1, D_in]
             # Concatenate new input
             
             # If step mode, x is [B, 1, D]
             pad_len = self.conv1d.kernel_size[0] - 1
             if prev_conv is None:
                 prev_conv = torch.zeros(batch, pad_len, self.d_inner, device=x.device)
                 
             conv_input = torch.cat([prev_conv, x_in], dim=1) # [B, K, D]
             # Run conv on window
             x_conv = self.conv1d(conv_input.transpose(1, 2)).transpose(1, 2)
             x_conv = x_conv[:, -1:, :] # Take only last output
             new_conv_state = conv_input[:, -pad_len:, :]
             
        x_conv = self.act(x_conv)
        x_ssm = x_conv
        
        # 3. SSM (Selective Scan)
        # Delta, B, C computation
        x_dbl = self.x_proj(x_ssm) # [B, L, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # [B, L, D_in]
        
        # Discretize A
        A = -torch.exp(self.A_log) # [D_in, D_state]
        
        # Parallel Scan (Training) or Recurrence (Inference)
        if seq_len > 1 and state is None:
            # Simplified Parallel Scan (Not real parallel assoc scan, just loop for baseline)
            # A real implementation would use a CUDA kernel or Associative Scan
            # For baseline correctness, we use a simple loop (slow but correct properties)
            y = []
            h = torch.zeros(batch, self.d_inner, self.A_log.shape[1], device=x.device)
            
            for t in range(seq_len):
                dt_t = dt[:, t, :].unsqueeze(-1) # [B, D_in, 1]
                dA = torch.exp(dt_t * A) # [B, D_in, D_state]
                
                x_t = x_ssm[:, t, :].unsqueeze(-1) # [B, D_in, 1]
                B_t = B[:, t, :].unsqueeze(1) # [B, 1, D_state]
                
                # B' = (dt * x) * B
                dB = (dt_t * x_t) * B_t # [B, D_in, D_state]
                
                h = h * dA + dB
                
                y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1) # [B, D_in]
                y.append(y_t)
                
            y = torch.stack(y, dim=1) # [B, L, D_in]
            y = y + x_ssm * self.D
            new_ssm_state = h
            new_conv_state = None # Not tracking for training return
            
        else:
            # Step Mode (L=1)
            h = state[1] if state and state[1] is not None else torch.zeros(batch, self.d_inner, self.A_log.shape[1], device=x.device)
            
            # Dimensions
            # dt: [B, 1, D_in]
            dt_t = dt.unsqueeze(-1) # [B, 1, D_in, 1]
            dA = torch.exp(dt_t * A) # [B, 1, D_in, D_state]
            
            # B: [B, 1, D_state]
            B_t = B.unsqueeze(2) # [B, 1, 1, D_state]
            x_t = x_ssm.unsqueeze(-1) # [B, 1, D_in, 1]
            
            # Recurrence
            # h: [B, D_in, D_state] -> [B, 1, D_in, D_state]
            h = h.unsqueeze(1) * dA + B_t * x_t
            h = h.squeeze(1)
            
            # Output
            # C: [B, 1, D_state]
            y = (h * C.unsqueeze(2)).sum(dim=-1) # [B, D_in]
            y = y + x_ssm.squeeze(1) * self.D
            y = y.unsqueeze(1)
            
            new_ssm_state = h
        
        # 4. Multiply with gate
        out = y * self.act(z)
        out = self.out_proj(out)
        
        return out, (new_conv_state, new_ssm_state)
