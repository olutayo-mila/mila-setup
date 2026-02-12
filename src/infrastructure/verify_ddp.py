import torch
import os
import torch.distributed as dist

def verify_ddp():
    # 1. Read Environment Variables passed by torchrun
    # RANK: Global ID (0 = Boss, 1 = Worker)
    # LOCAL_RANK: ID on this specific machine (0 or 1)
    # WORLD_SIZE: Total number of GPUs (2)
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    # 2. Initialize the "Phone Line" (Process Group)
    # This is the moment they try to shake hands.
    try:
        dist.init_process_group(backend="nccl")
        print(f"👋 Rank {rank}/{world_size} (Local {local_rank}) - Connection initialized!")
    except Exception as e:
        print(f"❌ Rank {rank} failed to connect: {e}")
        return

    # 3. Set the specific GPU for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 4. The Real Test: All-Reduce
    # We create a tensor with value [1] on every GPU
    tensor = torch.ones(1).to(device)
    
    # We ask NCCL to sum them up across all GPUs
    # If successful, every GPU should now have value [2] (1 + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    result = tensor.item()
    if result == world_size:
        print(f"✅ Rank {rank}: Math works! Sum is {result} (Expected {world_size})")
    else:
        print(f"❌ Rank {rank}: Math failed! Sum is {result} (Expected {world_size})")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    verify_ddp()