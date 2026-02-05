import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os

# Define the relative path to conf
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"🚀 Running Experiment: {cfg.infrastructure.name}")
    print(f"📂 Reading Data from: {cfg.paths.data}")
    print(f"🔧 Device: {cfg.device}")
    
    # 1. Device Setup
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  WARNING: Config asks for CUDA but it's not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    # 2. Hyperparameters (from Hydra)
    print(f"⚙️  Batch Size: {cfg.batch_size}")
    print(f"⚙️  Learning Rate: {cfg.lr}")
    
    # 3. Placeholder for Real Work
    # model = instantiate(cfg.model)
    # train_loader = ...
    
    print("✅ Job Completed Successfully")

if __name__ == "__main__":
    main()