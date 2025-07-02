#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=128g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_5_%j.out
#SBATCH --error=logs/batch_5_%j.err

echo "Batch 5 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: normalize numel ones onesLike outer pad pad_sequence pairwise_distance pinverse pixel_shuffle poisson poisson_nll_loss polar polygamma pow prelu prod promote_types qr rad2deg rand rand_like randperm range ravel real reciprocal relu remainder repeat_interleave reshape result_type roll rot90 round rrelu rsqrt scatter scatter_add searchsorted selu_ set_flush_denormal sgn sigmoid sign signbit silu sin sinh slogdet softmax_ softmin_ softplus softshrink_ softsign solve sort sparse_coo_tensor split sqrt square squeeze stack std"

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
APIS=(normalize numel ones onesLike outer pad pad_sequence pairwise_distance pinverse pixel_shuffle poisson poisson_nll_loss polar polygamma pow prelu prod promote_types qr rad2deg rand rand_like randperm range ravel real reciprocal relu remainder repeat_interleave reshape result_type roll rot90 round rrelu rsqrt scatter scatter_add searchsorted selu_ set_flush_denormal sgn sigmoid sign signbit silu sin sinh slogdet softmax_ softmin_ softplus softshrink_ softsign solve sort sparse_coo_tensor split sqrt square squeeze stack std)
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

echo "Batch 5 completed: $(date)"
echo "Failed: $failed/64"
exit $failed
