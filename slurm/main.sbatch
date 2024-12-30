#!/bin/bash
#SBATCH --job-name=VRL
#SBATCH --output=slurm/logs/example_%j.out
#SBATCH --error=slurm/logs/example_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100
#SBATCH --mem=32G

# Debug information
echo "Running on node(s): $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Using GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA Version: $(nvcc --version | grep release)"

# Run the SLURM job
srun \
    --container-image nvcr.io#nvidia/pytorch:23.08-py3 \
    --container-mounts ${PWD}:${PWD} \
    --container-workdir ${PWD} \
    bash -c 'pip install -r requirements.txt; python3 example.py'
