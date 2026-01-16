import torch
import torch.nn as nn

def measure_peak_memory(model, func_call, *args, **kwargs):
    """
    Measures the peak VRAM usage of a model execution.
    
    Args:
        model: The PyTorch model (acts as context).
        func_call: A function or lambda that executes the forward pass.
                   Example: lambda: model(x)
        *args, **kwargs: Arguments passed to func_call (optional usage).
        
    Returns:
        peak_mb (float): Peak memory allocated during execution in MB.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # Ensure we are not tracking gradients if not needed, but typically benchmarks measure
        # training (with grads) or inference (no grads).
        # We assume the caller handles torch.no_grad() contexts if desired.
        
        try:
             func_call()
        except Exception as e:
            # We don't want to crash benchmark just because of OOM, usually handled by caller,
            # but we re-raise to be safe or return OOM indicator.
            raise e
            
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024 ** 2)
        return peak_mb
    else:
        # Fallback for CPU (not really "VRAM" but can't measure typically same way)
        return 0.0

def get_model_size_mb(model):
    """Returns model parameter size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
