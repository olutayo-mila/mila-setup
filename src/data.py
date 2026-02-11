import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import os

def get_dataloaders(cfg: DictConfig):
    """
    Creates DataLoaders for CIFAR10.
    Expects the data to be at cfg.dataset.root.
    """
    
    # 1. Expand shell variables (like $SLURM_TMPDIR) if Python sees them
    data_root = os.path.expandvars(cfg.dataset.root)
    print(f"💿 Data Module loading from: {data_root}")

    # 2. Define Standard CIFAR10 Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize with standard CIFAR10 mean/std
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 3. Load Datasets
    # download=True is SAFE here because:
    # A) On Cluster: We pre-copied the .tar.gz file. Torchvision sees it and extracts it (no download).
    # B) On Laptop: It actually downloads it from the internet (convenient).
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory
    )

    return train_loader, val_loader