#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=128g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_4_%j.out
#SBATCH --error=logs/batch_4_%j.err

echo "Batch 4 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: leaky_relu lerp lgamma linear_ linspace log log10 log1p log2 logSoftmaxClass log_softmax logaddexp logaddexp2 logcumsumexp logdet logical_and logical_not logical_or logical_xor logit logsigmoid logspace logsumexp lp_pool1d_ lp_pool2d lstsq lt lu_solve lu_unpack margin_ranking_loss masked_select matmul matrix_exp matrix_power matrix_rank max max_pool1d max_pool2d max_pool3d max_unpool2d maximum mean median min minimum mm movedim mse_loss msort mul multiheadAttentionClass multilabel_soft_margin_loss multinomial mv mvlgamma nanmedian nansum narrow ne neg nextafter nll_loss nonzero norm"

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
APIS=(leaky_relu lerp lgamma linear_ linspace log log10 log1p log2 logSoftmaxClass log_softmax logaddexp logaddexp2 logcumsumexp logdet logical_and logical_not logical_or logical_xor logit logsigmoid logspace logsumexp lp_pool1d_ lp_pool2d lstsq lt lu_solve lu_unpack margin_ranking_loss masked_select matmul matrix_exp matrix_power matrix_rank max max_pool1d max_pool2d max_pool3d max_unpool2d maximum mean median min minimum mm movedim mse_loss msort mul multiheadAttentionClass multilabel_soft_margin_loss multinomial mv mvlgamma nanmedian nansum narrow ne neg nextafter nll_loss nonzero norm)
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

echo "Batch 4 completed: $(date)"
echo "Failed: $failed/64"
exit $failed
