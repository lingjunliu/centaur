#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=128g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_6_%j.out
#SBATCH --error=logs/batch_6_%j.err

echo "Batch 6 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: std_mean sub sum svd symeig t tan tanh_ tensordot topk transpose trapz triangular_solve tril tril_indices_ triu triu_indices trunc unbind unique unique_consecutive unsqueeze var var_mean where zeros zeros_like arctan_ UninitializedParameter get_default_device tile is_autocast_cache_enabled log_ log_ndtr view_as_complex_copy take is_autocast_enabled swapdims lu crow_indices_copy threshold fft iinfo celu set_deterministic_debug_mode ndtr spherical_bessel_j0 softmax negative_ abs_ arccos rsqrt_ rfft2 frexp gammainc Error finfo multi_dot ConstantPad1d cos_ stft i1 empty optimized_execution"

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

echo "Starting 64 APIs in parallel..."

# Run all APIs in parallel
APIS=(std_mean sub sum svd symeig t tan tanh_ tensordot topk transpose trapz triangular_solve tril tril_indices_ triu triu_indices trunc unbind unique unique_consecutive unsqueeze var var_mean where zeros zeros_like arctan_ UninitializedParameter get_default_device tile is_autocast_cache_enabled log_ log_ndtr view_as_complex_copy take is_autocast_enabled swapdims lu crow_indices_copy threshold fft iinfo celu set_deterministic_debug_mode ndtr spherical_bessel_j0 softmax negative_ abs_ arccos rsqrt_ rfft2 frexp gammainc Error finfo multi_dot ConstantPad1d cos_ stft i1 empty optimized_execution)
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

echo "Batch 6 completed: $(date)"
echo "Failed: $failed/64"
exit $failed
