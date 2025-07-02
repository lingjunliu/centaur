#!/bin/bash
module load cuda/12.4.0
spack load python@3.12.5 
spack load py-pip@23.1.2 ^python@3.12.5
source venv/bin/activate

echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID" 
echo "Time: $(date)"

# Project environment variables
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export PYTHONWARNINGS="ignore"

# Tensorflow environment variables  
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# GPU environment
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

echo "Environment configured:"
echo "  APIS_PER_NODE: $APIS_PER_NODE"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd $PROJECT_DIR

# Verify environment
echo "Python version: $(python --version)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create necessary directories
mkdir -p logs .tmp

# python scripts/scheduler.py torch

echo "Node job completed: $(date)"
