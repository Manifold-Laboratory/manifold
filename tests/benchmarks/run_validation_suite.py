"""
Manifold Validation Suite
=========================
Master runner that executes all validation benchmarks and generates summary report.

Usage:
    python tests/benchmarks/run_validation_suite.py
    
    # Run specific benchmarks:
    python tests/benchmarks/run_validation_suite.py --only overhead scaling
"""

import subprocess
import sys
import json
import argparse
import os
from pathlib import Path
from datetime import datetime


# Fix path calculation - go up from tests/benchmarks to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "tests/benchmarks/core"
RESULTS_DIR = PROJECT_ROOT / "tests/benchmarks/results/validation"


BENCHMARKS = {
    'overhead': {
        'script': 'benchmark_overhead.py',
        'description': 'Feature overhead analysis (VRAM, throughput)',
        'outputs': ['overhead_analysis.json', 'overhead_comparison.png']
    },
    'ablation': {
        'script': 'benchmark_feature_ablation.py',
        'description': 'Feature ablation (does each feature help?)',
        'outputs': ['ablation_results.json', 'ablation_chart.png']
    },
    'scaling': {
        'script': 'benchmark_scaling.py',
        'description': 'Scaling laws (params vs speed/memory)',
        'outputs': ['scaling_laws.json', 'scaling_curves.png']
    },
    'baseline': {
        'script': 'benchmark_baseline_comparison.py',
        'description': 'Baseline comparison (vs GRU/LSTM)',
        'outputs': ['baseline_comparison.json', 'baseline_chart.png']
    },
    'needle': {
        'script': 'benchmark_needle_haystack.py',
        'description': 'Long-context O(1) memory test',
        'outputs': ['../long_context/vram_vs_context.png']
    },
    'integrators': {
        'script': 'benchmark_integrators.py',
        'description': 'Numerical integrator comparison',
        'outputs': ['../integrators/integrator_metrics.json']
    }
}


def run_benchmark(name, info):
    """Run a single benchmark using direct script execution."""
    script_path = BENCHMARK_DIR / info['script']
    
    print(f"\n{'='*60}")
    print(f"[*] Running: {name.upper()}")
    print(f"   {info['description']}")
    print(f"{'='*60}")
    
    if not script_path.exists():
        print(f"[*] Script not found: {script_path}")
        return False
    
    try:
        # Use direct script execution with PYTHONPATH set
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per benchmark
        )
        
        print(result.stdout)
        if result.stderr and result.returncode != 0:
            # Only show stderr on failure
            print("STDERR:", result.stderr[-1000:])
        
        success = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("[*] TIMEOUT (10 minutes)")
        success = False
    except Exception as e:
        print(f"[*] ERROR: {e}")
        success = False
    
    return success


def generate_summary_report():
    """Generate summary report from all results."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }
    
    # Load available results
    for name, info in BENCHMARKS.items():
        for output in info['outputs']:
            if output.endswith('.json'):
                json_path = RESULTS_DIR / output
                if json_path.exists():
                    with open(json_path) as f:
                        report['benchmarks'][name] = json.load(f)
                    break
    
    # Save summary
    with open(RESULTS_DIR / "validation_summary.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("[*] VALIDATION SUMMARY")
    print("="*60)
    
    # Overhead summary
    if 'overhead' in report['benchmarks']:
        overhead = report['benchmarks']['overhead']
        if 'baseline' in overhead and 'all_features' in overhead:
            base = overhead['baseline']
            full = overhead['all_features']
            print(f"\n[*] OVERHEAD:")
            print(f"   Baseline: {base.get('vram_mb', '?')} MB, {base.get('throughput', '?')} seq/s")
            print(f"   All Features: {full.get('vram_mb', '?')} MB, {full.get('throughput', '?')} seq/s")
    
    # Ablation summary
    if 'ablation' in report['benchmarks']:
        ablation = report['benchmarks']['ablation']
        print(f"\n[*] FEATURE ABLATION:")
        for config, metrics in ablation.items():
            if 'accuracy' in metrics:
                print(f"   {config}: {metrics['accuracy']}% accuracy")
    
    # Baseline summary
    if 'baseline' in report['benchmarks']:
        baseline = report['benchmarks']['baseline']
        print(f"\n[*]️  BASELINE COMPARISON:")
        for task, results in baseline.items():
            if 'Manifold' in results and 'accuracy' in results['Manifold']:
                manifold_acc = results['Manifold']['accuracy']
                gru_acc = results.get('GRU', {}).get('accuracy', '?')
                print(f"   {task}: Manifold {manifold_acc}% vs GRU {gru_acc}%")
    
    print(f"\n[*] Full report saved to: {RESULTS_DIR / 'validation_summary.json'}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Run Manifold Validation Suite')
    parser.add_argument('--only', nargs='+', choices=list(BENCHMARKS.keys()),
                       help='Run only specific benchmarks')
    args = parser.parse_args()
    
    print("="*60)
    print("[*] MANIFOLD VALIDATION SUITE")
    print("="*60)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Results: {RESULTS_DIR}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine which benchmarks to run
    to_run = args.only if args.only else list(BENCHMARKS.keys())
    
    print(f"\n[*] Benchmarks to run: {', '.join(to_run)}")
    
    # Run benchmarks
    results = {}
    for name in to_run:
        if name in BENCHMARKS:
            success = run_benchmark(name, BENCHMARKS[name])
            results[name] = 'SUCCESS' if success else 'FAILED'
    
    # Summary
    print("\n" + "="*60)
    print("[*] EXECUTION RESULTS")
    print("="*60)
    for name, status in results.items():
        emoji = '[*]' if status == 'SUCCESS' else '[*]'
        print(f"   {emoji} {name}: {status}")
    
    # Generate summary report
    generate_summary_report()


if __name__ == "__main__":
    main()
