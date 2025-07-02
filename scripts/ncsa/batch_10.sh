#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=128g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_10_%j.out
#SBATCH --error=logs/batch_10_%j.err

echo "Batch 10 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: set_autocast_cache_enabled row_stack adjoint get_num_threads divide sparse_bsr_tensor swapaxes conj_physical_ are_deterministic_algorithms_enabled is_anomaly_enabled Mish BCELoss select quantile vsplit rsub logit_ set_num_interop_threads native_dropout spmm is_tracing sigmoid_ arccosh histogram ndtri ldexp ones_like is_scripting vitals_enabled hann_window set_autocast_ipu_dtype erfc_ sinc modified_bessel_k0 typename tan_ isinstance count_nonzero SoftMarginLoss set_default_dtype ParameterList autocast fix Hardsigmoid channel_shuffle CosineEmbeddingLoss acos_ is_floating_point bitwise_right_shift sym_fresh_size clip_ Identity bessel_y1 trunc_ bartlett_window index_copy clip ZeroPad2d set_num_threads DataParallel any vector_norm LazyBatchNorm2d clone"

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
APIS=(set_autocast_cache_enabled row_stack adjoint get_num_threads divide sparse_bsr_tensor swapaxes conj_physical_ are_deterministic_algorithms_enabled is_anomaly_enabled Mish BCELoss select quantile vsplit rsub logit_ set_num_interop_threads native_dropout spmm is_tracing sigmoid_ arccosh histogram ndtri ldexp ones_like is_scripting vitals_enabled hann_window set_autocast_ipu_dtype erfc_ sinc modified_bessel_k0 typename tan_ isinstance count_nonzero SoftMarginLoss set_default_dtype ParameterList autocast fix Hardsigmoid channel_shuffle CosineEmbeddingLoss acos_ is_floating_point bitwise_right_shift sym_fresh_size clip_ Identity bessel_y1 trunc_ bartlett_window index_copy clip ZeroPad2d set_num_threads DataParallel any vector_norm LazyBatchNorm2d clone)
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

echo "Batch 10 completed: $(date)"
echo "Failed: $failed/64"
exit $failed
