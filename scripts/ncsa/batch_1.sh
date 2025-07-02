#!/bin/bash
#SBATCH --job-name=orcl_torch_batch_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=128g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_1_%j.out
#SBATCH --error=logs/batch_1_%j.err

echo "Batch 1 starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: Flatten FractionalMaxPool2d GroupNorm Hardshrink Hardswish_ Hardtanh InstanceNorm1d InstanceNorm2d InstanceNorm3d L1Loss LPPool1d LPPool2d LSTMCell LayerNorm LeakyReLU Linear LogSigmoid_ MSELoss MarginRankingLoss MaxPool2d MaxPool3d MaxUnpool2d MultiLabelSoftMarginLoss MultiMarginLoss NLLLoss PReLU_ PairwiseDistance PixelShuffle PoissonNLLLoss ReLU6_ ReLU_ ReflectionPad1d ReflectionPad2d ReplicationPad1d ReplicationPad3d SELU SiLU_ Sigmoid_ Softmax Softmax2d Softmin Softplus_ Softshrink Softsign_ abs acos acosh adaptive_avg_pool1d adaptive_avg_pool2d adaptive_max_pool1d adaptive_max_pool2d add addbmm addcdiv addcmul addmm addmv addr allclose alpha_dropout amax amin angle arange"

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
APIS=(Flatten FractionalMaxPool2d GroupNorm Hardshrink Hardswish_ Hardtanh InstanceNorm1d InstanceNorm2d InstanceNorm3d L1Loss LPPool1d LPPool2d LSTMCell LayerNorm LeakyReLU Linear LogSigmoid_ MSELoss MarginRankingLoss MaxPool2d MaxPool3d MaxUnpool2d MultiLabelSoftMarginLoss MultiMarginLoss NLLLoss PReLU_ PairwiseDistance PixelShuffle PoissonNLLLoss ReLU6_ ReLU_ ReflectionPad1d ReflectionPad2d ReplicationPad1d ReplicationPad3d SELU SiLU_ Sigmoid_ Softmax Softmax2d Softmin Softplus_ Softshrink Softsign_ abs acos acosh adaptive_avg_pool1d adaptive_avg_pool2d adaptive_max_pool1d adaptive_max_pool2d add addbmm addcdiv addcmul addmm addmv addr allclose alpha_dropout amax amin angle arange)
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

echo "Batch 1 completed: $(date)"
echo "Failed: $failed/64"
exit $failed
