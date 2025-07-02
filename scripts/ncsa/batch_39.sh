#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_39
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_39_%j.out
#SBATCH --error=logs/batch_39_%j.err

echo "Batch 39 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: sinc modified_bessel_k0 typename tan_ isinstance count_nonzero SoftMarginLoss set_default_dtype ParameterList autocast fix Hardsigmoid channel_shuffle CosineEmbeddingLoss acos_ is_floating_point"

# Load environment
cd /projects/bdfv/aqin/dll-fuzzing-with-input-invariants
module load cuda/12.4.0
spack load python@3.12.5 
spack load py-pip@23.1.2 ^python@3.12.5
source venv/bin/activate

# Set environment
export PYTHONPATH=/projects/bdfv/aqin/dll-fuzzing-with-input-invariants:$PYTHONPATH
export PYTHONWARNINGS="ignore"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0


mkdir -p .tmp/oracle_results_torch

echo "Starting 16 APIs in parallel..."

# Run all APIs in parallel
APIS=(sinc modified_bessel_k0 typename tan_ isinstance count_nonzero SoftMarginLoss set_default_dtype ParameterList autocast fix Hardsigmoid channel_shuffle CosineEmbeddingLoss acos_ is_floating_point)
pids=()

for api in "${APIS[@]}"; do
    python -m eval.crash_monitor "$api" "torch" -1 -1 &
    pids+=($!)
done

# Wait for all to complete
failed=0
for pid in "${pids[@]}"; do
    wait $pid || ((failed++))
done

echo "Batch 39 completed: $(date)"
echo "Failed: $failed/16"
exit $failed
