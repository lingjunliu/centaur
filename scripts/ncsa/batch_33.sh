#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_33
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_33_%j.out
#SBATCH --error=logs/batch_33_%j.err

echo "Batch 33 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: is_storage fftshift LazyInstanceNorm2d script_if_tracing greater autocast_decrement_nesting meshgrid hermite_polynomial_he no_grad miopen_batch_norm fmax RReLU get_total_norm moveaxis ScriptWarning from_numpy"

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
APIS=(is_storage fftshift LazyInstanceNorm2d script_if_tracing greater autocast_decrement_nesting meshgrid hermite_polynomial_he no_grad miopen_batch_norm fmax RReLU get_total_norm moveaxis ScriptWarning from_numpy)
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

echo "Batch 33 completed: $(date)"
echo "Failed: $failed/16"
exit $failed
