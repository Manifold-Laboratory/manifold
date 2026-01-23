"""
Professional Visualization Suite Runner
========================================
Unified execution engine for all Manifold GFN visualizations.
Runs the complete suite and generates a consolidated results report.

Usage:
    python tests/benchmarks/viz/run_viz_suite.py --checkpoint checkpoints/model.pt
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
import tabulate # Assuming it's available or use simple formatting

# Config
VIZ_DIR = Path(__file__).parent
PROJECT_ROOT = VIZ_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_script(script_path, checkpoint=None):
    """Executes a single visualization script."""
    cmd = [sys.executable, str(script_path)]
    if checkpoint:
        cmd.append(checkpoint)
    
    start_time = time.time()
    try:
        # Run and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            cwd=str(PROJECT_ROOT)
        )
        elapsed = time.time() - start_time
        return True, result.stdout, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        return False, e.stderr, elapsed

def main():
    parser = argparse.ArgumentParser(description="Manifold GFN Visualization Suite Runner")
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--filter', type=str, help='Regex filter for script names')
    parser.add_argument('--skip-failures', action='store_true', help='Continue even if a script fails')
    
    args = parser.parse_args()
    
    # 1. Clear Screen & Header
    print("\033[H\033[J") # Clear terminal
    print("=" * 80)
    print("  🌊 MANIFOLD GFN: PROFESSIONAL VISUALIZATION SUITE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Viz Directory: {VIZ_DIR}")
    print(f"Checkpoint: {args.checkpoint or 'RANDOM WEIGHTS (Structural Mode)'}")
    print("-" * 80)

    # 2. Identify Scripts
    scripts = sorted([f for f in VIZ_DIR.glob("vis_*.py")])
    if args.filter:
        import re
        scripts = [s for s in scripts if re.search(args.filter, s.name)]

    print(f"Found {len(scripts)} visualization modules.")
    
    # 3. Execution Loop
    results = []
    for i, script in enumerate(scripts):
        rel_path = script.relative_to(PROJECT_ROOT)
        print(f"[{i+1}/{len(scripts)}] Running {script.name}...", end="", flush=True)
        
        success, output, elapsed = run_script(script, args.checkpoint)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f" {status} ({elapsed:.1f}s)")
        
        results.append({
            "Module": script.name,
            "Status": status,
            "Runtime": f"{elapsed:.1f}s",
            "Output": "Success" if success else output.strip().split('\n')[-1]
        })
        
        if not success and not args.skip_failures:
            print(f"\n🛑 CRITICAL FAILURE in {script.name}")
            print("-" * 40)
            print(output)
            print("-" * 40)
            sys.exit(1)

    # 4. Final Report Table
    print("\n" + "=" * 80)
    print("  SUITE EXECUTION SUMMARY")
    print("=" * 80)
    
    header = ["Module", "Status", "Runtime", "Last Message"]
    fmt = "{:<25} | {:<8} | {:<8} | {:<30}"
    print(fmt.format(*header))
    print("-" * 80)
    
    for r in results:
        msg = r["Output"]
        if len(msg) > 30: msg = msg[:27] + "..."
        print(fmt.format(r["Module"], r["Status"], r["Runtime"], msg))
    
    print("-" * 80)
    print(f"Successfully generated {sum(1 for r in results if 'PASS' in r['Status'])} visualizations.")
    print(f"Results located in: tests/results/viz/")
    print("=" * 80)

if __name__ == "__main__":
    main()
