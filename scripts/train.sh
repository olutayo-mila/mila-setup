#!/bin/bash
#SBATCH --job-name=cifar-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

# 1. Setup the Environment
module load python/3.10
source $HOME/projects/mila-setup/.venv/bin/activate

# 2. Define locations
SOURCE_DATA="/network/datasets/cifar10/cifar-10-python.tar.gz"
DEST_DIR="$SLURM_TMPDIR/cifar10"

# 3. Create destination and Copy
echo "Step 1: Creating directory $DEST_DIR"
mkdir -p $DEST_DIR

echo "Step 2: Copying data from Network to Local SSD..."
cp $SOURCE_DATA $DEST_DIR/

# 4. Run Python
# Note: torchvision will see the .tar.gz file and automatically extract it
# strictly BECAUSE we set download=True (or allow it to verify).
echo "Step 3: Launching Training..."
uv run python -m src.train infrastructure=cluster