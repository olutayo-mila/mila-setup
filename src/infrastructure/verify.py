import torch
import os
import sys

def verify_setup():
    print("=== 🔍 Environment Verification ===")
    
    # 1. Check Python Location (Are we in the venv?)
    print(f"🐍 Python Path: {sys.executable}")
    
    # 2. Check CUDA
    print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   Device Count: {torch.cuda.device_count()}")
    
    # 3. Check Important Slurm Variables
    print("\n=== ⚡ Slurm Variables ===")
    keys = ["SLURM_JOB_ID", "SLURM_TMPDIR", "SLURM_NODELIST", "SLURM_NTASKS"]
    for key in keys:
        print(f"   {key}: {os.environ.get(key, 'NOT SET')}")

    # 4. Check Data Directory Access
    # If on cluster, check if $SLURM_TMPDIR is writable
    tmp_dir = os.environ.get("SLURM_TMPDIR")
    if tmp_dir:
        print(f"\n=== 📂 Storage Check ===")
        print(f"   $SLURM_TMPDIR is at: {tmp_dir}")
        if os.access(tmp_dir, os.W_OK):
            print("   ✅ Write permission verified.")
        else:
            print("   ❌ NO WRITE PERMISSION!")

if __name__ == "__main__":
    verify_setup()