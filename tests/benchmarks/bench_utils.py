import torch
import torch.nn as nn
import json
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ResultsLogger:
    """
    Standardized logger for benchmark results.
    Handles path management, JSON serialization, and publication-quality plotting.
    """
    def __init__(self, benchmark_name, category="core"):
        self.benchmark_name = benchmark_name
        self.category = category
        self.root = Path(__file__).parent.parent
        self.results_dir = self.root / "results" / category / benchmark_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting aesthetics
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def save_json(self, data, filename="metrics.json"):
        """Saves metadata and results to a JSON file."""
        output = {
            "benchmark": self.benchmark_name,
            "category": self.category,
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "torch_version": torch.__version__
            },
            "data": data
        }
        path = self.results_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)
        print(f"✅ JSON results saved to: {path}")
        return path

    def save_plot(self, fig, filename):
        """Saves a matplotlib figure with publication quality."""
        path = self.results_dir / filename
        fig.savefig(path)
        plt.close(fig)
        print(f"✅ Plot saved to: {path}")
        return path

class PerformanceStats:
    """Utilities for accurate performance measurement."""
    
    @staticmethod
    def get_model_size_mb(model):
        """Returns model parameter + buffer size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024**2

    @staticmethod
    def measure_peak_memory(model, func_call, *args, **kwargs):
        """Measures peak VRAM usage during a function call."""
        if not torch.cuda.is_available():
            return 0.0
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            func_call(*args, **kwargs)
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            return peak_bytes / 1024**2
        except Exception as e:
            print(f"⚠️ Memory measurement error: {e}")
            raise e

def measure_peak_memory(model, func_call, *args, **kwargs):
    """Legacy wrapper for backward compatibility."""
    return PerformanceStats.measure_peak_memory(model, func_call, *args, **kwargs)

def get_model_size_mb(model):
    """Legacy wrapper for backward compatibility."""
    return PerformanceStats.get_model_size_mb(model)
