import unittest
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_suite():
    print("=" * 70)
    print("      GFN  TEST SUITE ")
    print("=" * 70)
    print("\nFor comprehensive report with visualizations, run:")
    print("  python tests/benchmarks/generate_report.py --checkpoint your_model.pt")
    print("\n" + "=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 1. Discover Unit Tests
    start_dir = str(Path(__file__).parent)
    print(f"\n📦 Discovery in: {start_dir}")
    
    # Unit Tests
    suite.addTests(loader.discover(start_dir + '/unit', pattern='test_*.py'))
    suite.addTests(loader.discover(start_dir + '/integration', pattern='test_*.py'))
    suite.addTests(loader.discover(start_dir + '/physics', pattern='test_*.py'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        print("\n❌ [FAIL] UNIT TESTS FAILED")
        sys.exit(1)
    else:
        print("\n✅ [PASS] UNIT TESTS PASSED")

    # 2. Run Script-based Tests
    import subprocess
    
    scripts = [
        ("tests/integration/test_overfit_sanity.py", "Overfit Diagnosis (Sanity Check)")
    ]
    
    print("\n" + "=" * 70)
    print("      RUNNING INTEGRATION SCRIPTS")
    print("=" * 70)
    
    all_passed = True
    
    for script_rel, name in scripts:
        script_path = PROJECT_ROOT / script_rel
        
        # Skip if doesn't exist
        if not script_path.exists():
            print(f"\n⚠️  Skipping: {name} (file not found)")
            continue
        
        print(f"\n▶ Running: {name} ({script_rel})...", flush=True)
        
        try:
            ret = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if ret.returncode == 0:
                print(f"✅ [PASS] {name}")
                if ret.stdout:
                    last_line = ret.stdout.strip().splitlines()[-1] if ret.stdout.strip() else ''
                    if last_line:
                        print(f"   Last output: {last_line}")
            else:
                print(f"❌ [FAIL] {name}")
                print(f"   Error Output:\n{ret.stderr[:500]}")  # Limit output
                all_passed = False
        except subprocess.TimeoutExpired:
            print(f"⏱️  [TIMEOUT] {name}")
            all_passed = False
        except Exception as e:
            print(f"💥 [CRITICAL ERROR] {e}")
            all_passed = False
            
    print("\n" + "=" * 70)
    if all_passed and result.wasSuccessful():
        print("✅ [ALL PASSED] CORE TESTS VERIFIED")
        print("\n💡 TIP: Run benchmarks for detailed analysis:")
        print("   python tests/benchmarks/generate_report.py")
    else:
        print("❌ [FAILED] SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    run_suite()
