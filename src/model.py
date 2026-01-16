import torch
import torch.nn as nn
from .layers import MLayer, ParallelMLayer


class Manifold(nn.Module):
    """
    MANIFOLD: Multi-scale Adaptive Neural Inference via Flow On Learned Dynamics
    
    A sequence model that evolves hidden states as geodesic flows
    on a Riemannian manifold. Achieves O(1) memory complexity.
    
    Architecture:
        1. Embedding: Token -> Force impulse on manifold
        2. Dynamics: M-Layers evolve state (x, v) via geodesic flow
        3. Readout: Position x -> Logits via learned projection
    
    Args:
        vocab_size: Size of vocabulary
        dim: Hidden dimension (default: 256)
        depth: Number of M-Layers (default: 4)
        rank: Low-rank Christoffel approximation (default: 32)
        heads: Number of independent geodesic heads (default: 4)
        integrator_type: 'heun', 'rk4', or 'symplectic' (default: 'heun')
    
    Example:
        >>> model = Manifold(vocab_size=16, dim=512, depth=12, integrator_type='heun')
        >>> logits, state = model(input_ids)
    """
    
    
    def __init__(self, vocab_size, dim=256, depth=4, rank=32, heads=4, integrator_type='heun', use_scan=False, physics_config=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.integrator_type = integrator_type
        self.use_scan = use_scan
        self.physics_config = physics_config or {}
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Stack of Multi-Head Manifold Layers
        print(f"[*] MANIFOLD Init: {depth} layers, {heads} heads, {dim} dim, {integrator_type}, scan={use_scan}")
        if self.physics_config.get('active_inference', {}).get('enabled', False):
             print(f"[*] Active Inference ENABLED (Plasticity, Singularities, Dynamic Time)")
        
        self.layers = nn.ModuleList()
        for _ in range(depth):
            if use_scan:
                self.layers.append(ParallelMLayer(dim, heads=heads, physics_config=self.physics_config))
            else:
                # v0.8.0 Fractal Manifolds
                if self.physics_config.get('fractal', {}).get('enabled', False):
                    from .layers import FractalMLayer
                    self.layers.append(FractalMLayer(dim, heads=heads, rank=rank, integrator_type=integrator_type, physics_config=self.physics_config))
                else:
                    self.layers.append(MLayer(dim, heads=heads, rank=rank, integrator_type=integrator_type, physics_config=self.physics_config))
        
        # Output projection
        self.readout_norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, vocab_size)
        
        # Improved Initialization (Critical for convergence)
        # Non-zero random init helps early gradient flow
        self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)

        # Init weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, state=None):
        """
        Forward pass through the geodesic flow.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len] (1=valid, 0=pad)
            state: Optional tuple (x, v) to continue from previous state
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            state: Final state tuple (x, v) for continuation
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize state from learnable parameters or provided state
        if state is None:
            x = self.x0.expand(batch_size, -1)
            v = self.v0.expand(batch_size, -1)
        else:
            x, v = state
        
        # Pre-compute all token embeddings (forces)
        all_forces = self.embedding(input_ids)  # [batch, seq_len, dim]
        
        if self.use_scan:
            # === PARALLEL MODE (SCAN) ===
            # Process entire sequence at once
            
            # Initial states (broadcast to batch)
            x = self.x0.expand(batch_size, seq_len, -1) # Wait, x0 is purely initial. 
            # In scan, x evolves. We pass 'x' through layers? 
            # In MLayer 'x' is input state.
            # But ParallelMLayer takes 'force' sequence and generates (x,v) sequence.
            # It's a bit different. The 'force' for layer L comes from readout of layer L-1?
            # Standard ResNet/Transformer: x_{l+1} = Layer(x_l)
            # Here: (x, v) = Layer(x, v, force)
            # For Parallel, we treat the input to the layer as the "Force" driving the flow.
            # So: Force_0 = Embedding(tokens)
            # (x_1, v_1) = Layer_1(Force_0)  <-- Parallel Scan
            # Force_1 = Projection(x_1) ?? Or just pass x_1 as force to next layer?
            # Let's say the conceptual "Force" for layer L is the output state 'x' of layer L-1.
            
            curr_input = all_forces # [B, L, D]
            all_christoffels = [] # To be populated if needed
            
            for layer in self.layers:
                # Parallel Layer takes full sequence of inputs
                # We treat the input 'curr_input' as the driving force sequence
                out_x, out_v, out_ctx, layer_christoffels = layer(None, None, force=curr_input)
                all_christoffels.extend(layer_christoffels)
                
                # Update input for next layer (stacking manifold layers)
                # Next layer is driven by the position/state of previous layer
                curr_input = out_x # Use position as input to next layer
                
            # Final Readout
            # [batch, seq_len, dim]
            x_final = curr_input 
            out = self.readout_norm(x_final)
            logits = self.readout(out) # [batch, seq_len, vocab_size]
            
            # Return last state for compatibility?
            # Just return zeros or last element
            return logits, (x_final[:, -1], None), all_christoffels

        else:
            # === SEQUENTIAL MODE (LOOP) ===
            # Prepare attention mask
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            else:
                mask = torch.ones(batch_size, seq_len, 1, device=input_ids.device)
            
            # Process sequence token by token (recurrent dynamics)
            logits_list = []
            all_christoffels = []
            
            for t in range(seq_len):
                # Get force for current timestep
                force = all_forces[:, t]  # [batch, dim]
                
                # Apply mask (zero force for padding tokens)
                force = force * mask[:, t]
                
                # Evolve state through all M-Layers
                context = None
                for layer in self.layers:
                    # Update state
                    x, v, context, layer_christoffels = layer(x, v, force, context)
                    # We store christoffels for the LAST token of the batch for regularization?
                    # Or all tokens? Usually loss is computed per token.
                    # To avoid memory explosion, we'll only return the current token's christoffels
                    # to be used in the current step calculation if needed.
                    all_christoffels = layer_christoffels # Keep last layer or all? Let's say all.
                
                # Readout: project position to vocabulary logits
                out = self.readout_norm(x)
                logit = self.readout(out)  # [batch, vocab_size]
                logits_list.append(logit.unsqueeze(1))
            
            # Stack all logits
            logits = torch.cat(logits_list, dim=1)  # [batch, seq_len, vocab_size]
            
            return logits, (x, v), all_christoffels
    
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None):
        """
        Autoregressive generation with sampling.
        
        Args:
            prompt_ids: Prompt token indices [1, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Softmax temperature (1.0 = normal, <1 = sharper)
            top_k: Limit to top K tokens (e.g. 40)
            top_p: Nucleus sampling probability (e.g. 0.9)
            
        Returns:
            generated_ids: Full sequence including prompt
        """
        self.eval()
        device = prompt_ids.device
        
        # Process prompt
        logits, state, _ = self(prompt_ids)
        
        # Start generation
        generated = prompt_ids.tolist()[0]
        
        def sample_next(logits, temp=1.0, k=None, p=None):
            # Last timestep logits
            next_logit = logits[:, -1, :] / temp
            probs = torch.softmax(next_logit, dim=-1)
            
            # Top-K
            if k is not None:
                v, _ = torch.topk(next_logit, k)
                next_logit[next_logit < v[:, [-1]]] = -float('Inf')
            
            # Top-P (Nucleus)
            if p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logit[indices_to_remove] = -float('Inf')
            
            # Sample
            if k is None and p is None:
                # Greedy
                return torch.argmax(next_logit, dim=-1, keepdim=True)
            else:
                # Multinomial
                probs = torch.softmax(next_logit, dim=-1)
                return torch.multinomial(probs, num_samples=1)

        # Initial sample
        curr_token = sample_next(logits, temperature, top_k, top_p)
        generated.append(curr_token.item())
        
        for _ in range(max_new_tokens - 1):
            logits, state, _ = self(curr_token, state=state)
            curr_token = sample_next(logits, temperature, top_k, top_p)
            generated.append(curr_token.item())
        
        return generated
