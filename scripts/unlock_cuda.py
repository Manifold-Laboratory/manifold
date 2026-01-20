
import os
import time
import shutil
from pathlib import Path

def cleanup_locks(target_file="gfn_cuda.pyd"):
    print(f"[*] Attempting to remove locked file: {target_file}")
    
    p = Path(target_file)
    if not p.exists():
        print("File does not exist. No action needed.")
        return

    # Retry loop
    max_retries = 5
    for i in range(max_retries):
        try:
            if p.exists():
                os.remove(p)
            print(f"✅ Successfully removed {target_file}")
            return
        except OSError as e:
            print(f"⚠️ Attempt {i+1}/{max_retries} failed: {e}")
            print("Waiting 2s for process to release handle...")
            time.sleep(2)
            
    print(f"❌ Could not remove {target_file}. It is strictly locked by another process (likely a suspended python.exe).")
    print("Please manually kill any python/pythonw benchmarks running in background.")

if __name__ == "__main__":
    cleanup_locks("gfn_cuda.pyd")
    # Also clean up any build temps
    if os.path.exists("build"):
        try:
            shutil.rmtree("build")
            print("✅ Removed build/ directory")
        except:
            pass
