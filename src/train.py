import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import wandb

# Import our new data module
from src.data import get_dataloaders

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"🚀 Running Experiment: {cfg.infrastructure.name}")
    
    # 1. Device Setup (Smart Auto-Detection)
    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        # User forced a specific device (e.g., "cuda:0")
        device = torch.device(cfg.device)

    # 2. Data Setup
    print("⬇️  Initializing Data...")
    train_loader, val_loader = get_dataloaders(cfg)
    print(f"✅ Data Loaded. Train batches: {len(train_loader)}")
    
    # 3. Model Setup (Placeholder for now)
    # model = instantiate(cfg.model).to(device)
    
    # 4. Init WandB
    if cfg.log_wandb:
        wandb.init(
            project="mila-experiment-1",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.infrastructure.name
        )
    # 5. Dummy Training Loop
    print("🔄 Starting Dummy Training Loop...")
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Just to prove it works
        if i == 0:
            print(f"   Batch Shape: {images.shape}")
            print(f"   Label Shape: {labels.shape}")

        # Fake a loss for testing
        fake_loss = 1.0 / (i + 1)
        
        if cfg.log_wandb:
            wandb.log({"train_loss": fake_loss, "batch": i})

        # Break early since this is just a test
        if i >= 5:
            break
            
    print("✅ Job Completed Successfully")

if __name__ == "__main__":
    main()