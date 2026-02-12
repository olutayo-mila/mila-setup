#!/bin/bash
#SBATCH --job-name=verify
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2        # Request 2 GPUs to test NCCL communication
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00

module load python/3.10
export UV_CACHE_DIR=$SCRATCH/.uv_cache

echo "Running Verification..."
# 'uv run' activates env automatically
uv run torchrun --nproc_per_node=2 src/infrastructure/verify.py