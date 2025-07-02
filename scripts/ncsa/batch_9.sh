#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=128g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_9_%j.out
#SBATCH --error=logs/batch_9_%j.err

echo "Batch 9 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: is_storage fftshift LazyInstanceNorm2d script_if_tracing greater autocast_decrement_nesting meshgrid hermite_polynomial_he no_grad miopen_batch_norm fmax RReLU get_total_norm moveaxis ScriptWarning from_numpy HuberLoss nan_to_num modified_bessel_i0 ReLU vstack Unflatten sspaddmm set_fusion_strategy is_warn_always_enabled eigvals get_deterministic_debug_mode gammaincc positive is_autocast_ipu_enabled select_copy arcsinh_ remove_weight_norm all parameters_to_vector Tanhshrink set_autocast_cpu_dtype fft2 ZeroPad1d scatter_reduce sym_float psi HingeEmbeddingLoss atan_ is_inference_mode_enabled greater_equal less vdot ceil_ CELU parse_type_comment enable_onednn_fusion hsplit LogSoftmax concatenate set_module bitwise_left_shift CompilationUnit get_rng_state SyncBatchNorm LazyBatchNorm1d Tanh view_as_real column_stack"

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
APIS=(is_storage fftshift LazyInstanceNorm2d script_if_tracing greater autocast_decrement_nesting meshgrid hermite_polynomial_he no_grad miopen_batch_norm fmax RReLU get_total_norm moveaxis ScriptWarning from_numpy HuberLoss nan_to_num modified_bessel_i0 ReLU vstack Unflatten sspaddmm set_fusion_strategy is_warn_always_enabled eigvals get_deterministic_debug_mode gammaincc positive is_autocast_ipu_enabled select_copy arcsinh_ remove_weight_norm all parameters_to_vector Tanhshrink set_autocast_cpu_dtype fft2 ZeroPad1d scatter_reduce sym_float psi HingeEmbeddingLoss atan_ is_inference_mode_enabled greater_equal less vdot ceil_ CELU parse_type_comment enable_onednn_fusion hsplit LogSoftmax concatenate set_module bitwise_left_shift CompilationUnit get_rng_state SyncBatchNorm LazyBatchNorm1d Tanh view_as_real column_stack)
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

echo "Batch 9 completed: $(date)"
echo "Failed: $failed/64"
exit $failed
