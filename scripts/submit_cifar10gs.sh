#!/bin/bash

#SBATCH --array=0-11
#SBATCH --job-name=cifar10gs
#SBATCH --output=logs/cifar10gs_%A_%a.out
#SBATCH --error=logs/cifar10gs_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nerfstudio

# Create logs directory
mkdir -p logs

# Determine if processing train or test set
if [ $SLURM_ARRAY_TASK_ID -lt 10 ]; then
    # Process training set (chunks 0-9)
    python create_cifar10gs_dataset.py --chunk_id $SLURM_ARRAY_TASK_ID --train
else
    # Process test set (chunks 10-11)
    python create_cifar10gs_dataset.py --chunk_id $((SLURM_ARRAY_TASK_ID - 10))
fi 