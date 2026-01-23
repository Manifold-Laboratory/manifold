"""
Standardized Validation Suite Runner
====================================

Master orchestrator for professional benchmarks:
- Discovers and runs core benchmarks.
- Aggregates metrics from standardized ResultsLogger outputs.
- Displays scientific performance summary.
"""

import subprocess
import sys
import json
import argparse
import os
from pathlib import Path
from datetime import datetime

# Path Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "tests/benchmarks/core"
RESULTS_DIR = PROJECT_ROOT / "tests/benchmarks/results/core"

BENCHMARKS = {
    'performance': {
        'script': 'benchmark_performance.py',
        'description': 'O(N^2) vs O(1) Memory & Scaling Analysis'
    },
    'integrators': {
        'script': 'benchmark_integrators.py',
        'description': 'Numerical Drift & Symplectic Stability'
    },
    'needle': {
        'script': 'benchmark_needle_haystack.py',
        'description': '1M Token Long-Context Stress Test'
    },
    'baseline': {
        'script': 'benchmark_baseline_comparison.py',
        'description': 'Systematic comparison vs RNNs (GRU/LSTM)'
    },
    'composition': {
        'script': 'benchmark_composition.py',
        'description': 'Function composition & systematic generalization'
    }
}

def run_benchmark(name, info):
    print(f"\n🚀 EXECUTING: {name.upper()}")
    print(f"   {info['description']}")
    print("-" * 60)
    
    script_path = BENCHMARK_DIR / info['script']
    if not script_path.exists():
        print(f"   ❌ Error: Script not found at {script_path}")
        return False

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=1200 # 20m for 1M context tests
        )
        
        if result.returncode == 0:
            print(f"   ✅ {name} completed successfully.")
            return True
        else:
            print(f"   ❌ {name} failed with exit code {result.returncode}")
            print(f"   Error: {result.stderr[-500:]}")
            return False
    except Exception as e:
        print(f"   ❌ Execution Error: {e}")
        return False

def print_master_summary():
    print("\n" + "="*80)
    print("📈 SCIENTIFIC VALIDATION SUMMARY")
    print("="*80)
    
    summary = {}
    
    for name in BENCHMARKS:
        metrics_path = RESULTS_DIR / name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                summary[name] = data["data"]
                
    # 1. Performance Summary
    if 'performance' in summary:
        print("\n[PERFORMANCE SCALING]")
        for item in summary['performance']:
            if item['Sequence Length'] in [1024, 8192, 16384]:
                print(f"   {item['Model']:20s} | L={item['Sequence Length']:5d} | VRAM: {item['VRAM (MB)']:8.1f} MB | {item['Throughput (seq/s)']:6.1f} seq/s")

    # 2. Integrator Summary
    if 'integrators' in summary:
        print("\n[PHYSICS STABILITY]")
        # Sort by drift
        sorted_integs = sorted(summary['integrators'], key=lambda x: x['Drift (%)'] if x['Drift (%)'] is not None else 1e9)
        for item in sorted_integs[:5]: # Top 5
            print(f"   {item['Integrator']:15s} | Drift: {item['Drift (%)']:10.6f}% | Speed: {item['Inference Speed']:6.1f} seq/s")

    # 3. Long Context Summary
    if 'needle' in summary:
        print("\n[LONG-CONTEXT (O(1) PROOF)]")
        needle = summary['needle']
        print(f"   Tested up to {needle[-1]['Sequence Length']} tokens")
        vram_start = needle[0]['VRAM (MB)']
        vram_end = needle[-1]['VRAM (MB)']
        increase = (vram_end - vram_start) / vram_start * 100
        print(f"   VRAM Expansion (1k -> {needle[-1]['Sequence Length']}): {increase:.2f}% (Target: <5%)")

    print("\n" + "="*80)
    print(f"✅ Master Report generated at: {RESULTS_DIR}")
    print(f"📂 Visualization results: {PROJECT_ROOT}/tests/benchmarks/results/core/")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Manifold Master Validation Runner')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--only', nargs='+', help='Specific benchmarks to run')
    parser.add_argument('--summary', action='store_true', help='Just show the summary of existing results')
    
    args = parser.parse_args()
    
    if args.summary:
        print_master_summary()
        return

    to_run = args.only if args.only else (BENCHMARKS.keys() if args.all else [])
    
    if not to_run:
        parser.print_help()
        return

    for name in to_run:
        if name in BENCHMARKS:
            run_benchmark(name, BENCHMARKS[name])
            
    print_master_summary()

if __name__ == "__main__":
    main()
