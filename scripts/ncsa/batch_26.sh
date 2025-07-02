#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_26
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_26_%j.out
#SBATCH --error=logs/batch_26_%j.err

echo "Batch 26 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: inv reciprocal_ PixelUnshuffle set_autocast_enabled sqrt_ enable_grad set_default_device svdvals BatchNorm2d put index_put conj_physical vander absolute dsplit modified_bessel_i1"

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
APIS=(inv reciprocal_ PixelUnshuffle set_autocast_enabled sqrt_ enable_grad set_default_device svdvals BatchNorm2d put index_put conj_physical vander absolute dsplit modified_bessel_i1)
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

echo "Batch 26 completed: $(date)"
echo "Failed: $failed/16"
exit $failed
