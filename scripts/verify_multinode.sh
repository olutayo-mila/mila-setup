#!/bin/bash
#SBATCH --job-name=ddp-multinode
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

# 1. Setup
module load python/3.10
export UV_CACHE_DIR=$SCRATCH/.uv_cache

# --- NETWORK FIXES ---
# Force PyTorch (GLOO) to use the main Ethernet interface for the handshake.
# We exclude loopback (lo), docker, and virtual bridges.
export GLOO_SOCKET_IFNAME=^lo,docker0,virbr0,vnet0

# Force NCCL (Data transfer) to behave similarly
export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0,vnet0

# Debugging: Print network interfaces so we can see what's happening if it fails
echo "=== Network Interfaces on Master Node ==="
ip addr show | grep inet
echo "========================================="
# ---------------------

# 2. Get Master Node Info
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

echo "Node List: ${nodes_array[@]}"
echo "Master Node Name: $head_node"

# --- THE MAGIC FIX ---
# Instead of using the hostname, we ask the head node for its primary IP address.
# `hostname -I` gives all IPs. `awk '{print $1}'` takes the first one (usually the main Ethernet).
master_addr=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
export MASTER_ADDR=$master_addr
echo "Master Node IP: $MASTER_ADDR"
# ---------------------

# 3. Launch
# We use the IP address ($MASTER_ADDR) for the rendezvous endpoint
srun uv run torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    src/infrastructure/verify_ddp.py