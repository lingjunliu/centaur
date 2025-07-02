#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=128g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_2_%j.out
#SBATCH --error=logs/batch_2_%j.err

echo "Batch 2 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: argmax argmin argsort as_strided as_tensor asin asinh atan atan2 atanh atleast_1d atleast_2d atleast_3d avg_pool1d avg_pool2d baddbmm batch_norm bernoulli binaryCrossEntropyWithLogits bincount bitwise_and bitwise_not bitwise_or bitwise_xor block_diag bmm broadcast_shapes broadcast_tensors broadcast_to bucketize cartesian_prod cat cdist ceil celu_ chain_matmul cholesky cholesky_inverse cholesky_solve chunk clamp clip_grad_norm_ combinations complex conj conv_transpose2d copysign cos cosh cosine_similarity countNonzero cross cross_entropy cummax cummin cumprod cumsum deg2rad det diag diag_embed diagflat diagonal digamma"

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
APIS=(argmax argmin argsort as_strided as_tensor asin asinh atan atan2 atanh atleast_1d atleast_2d atleast_3d avg_pool1d avg_pool2d baddbmm batch_norm bernoulli binaryCrossEntropyWithLogits bincount bitwise_and bitwise_not bitwise_or bitwise_xor block_diag bmm broadcast_shapes broadcast_tensors broadcast_to bucketize cartesian_prod cat cdist ceil celu_ chain_matmul cholesky cholesky_inverse cholesky_solve chunk clamp clip_grad_norm_ combinations complex conj conv_transpose2d copysign cos cosh cosine_similarity countNonzero cross cross_entropy cummax cummin cumprod cumsum deg2rad det diag diag_embed diagflat diagonal digamma)
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

echo "Batch 2 completed: $(date)"
echo "Failed: $failed/64"
exit $failed
