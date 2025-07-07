#!/usr/bin/env python3
"""
Simple Batch Processor - One script to rule them all
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def load_apis(project_dir, low=-1, high=-1):
    """Load APIs from file or range"""
    apis_file = "torch_apis.txt"

    if os.path.exists(apis_file):
        with open(apis_file, 'r') as f:
            apis = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(apis)} APIs from torch_apis.txt")
        return apis
    elif low != -1 and high != -1:
        apis = [str(i) for i in range(low, high + 1)]
        print(f"Generated {len(apis)} APIs from range {low}-{high}")
        return apis
    else:
        raise ValueError("No torch_apis.txt file found and no valid range specified")

def create_batch_script(batch_id, apis, lib, low, high, project_dir):
    """Create SBATCH script for batch"""
    api_list = " ".join(apis)
    
    script_content = f'''#!/bin/bash
#SBATCH --job-name=orcl_{lib}_batch_{batch_id}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64g
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfv-delta-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/batch_{batch_id}_%j.out
#SBATCH --error=logs/batch_{batch_id}_%j.err

echo "Batch {batch_id} starting: $(date)"
echo "Node: $SLURMD_NODENAME, Job: $SLURM_JOB_ID"
echo "APIs: {api_list}"

# Load environment
cd {project_dir}
module load cuda/12.4.0
spack load python@3.12.5 
spack load py-pip@23.1.2 ^python@3.12.5
source venv/bin/activate

# Set environment
export PYTHONPATH={project_dir}:$PYTHONPATH
export PYTHONWARNINGS="ignore"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0


mkdir -p .tmp/oracle_results_{lib}

echo "Starting {len(apis)} APIs in parallel..."

# Run all APIs in parallel
APIS=({api_list})
pids=()

for api in "${{APIS[@]}}"; do
    python -m eval.crash_monitor "$api" "{lib}" {low} {high} &
    pids+=($!)
done

# Wait for all to complete
failed=0
for pid in "${{pids[@]}}"; do
    wait $pid || ((failed++))
done

echo "Batch {batch_id} completed: $(date)"
echo "Failed: $failed/{len(apis)}"
exit $failed
'''
    
    # Write script
    script_path = f"{project_dir}/scripts/ncsa/batch_{batch_id}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description="Simple Batch Processor")
    parser.add_argument("lib", help="Library (torch/tf)")
    parser.add_argument("--low", type=int, default=-1)
    parser.add_argument("--high", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    
    args = parser.parse_args()
    project_dir = os.getcwd()  # Use current working directory
    
    # # Create directories
    # (project_dir / "logs").mkdir(exist_ok=True)
    # (project_dir / ".tmp").mkdir(exist_ok=True)
    
    # Load APIs and create batches
    apis = load_apis(project_dir, args.low, args.high)
    batches = [apis[i:i+args.batch_size] for i in range(0, len(apis), args.batch_size)]
    
    print(f"Creating {len(batches)} batches...")
    
    # # Create and submit batches
    for i, batch in enumerate(batches, 1):
        script_path = create_batch_script(i, batch, args.lib, args.low, args.high, project_dir)
        
    #     # Submit to SLURM
    #     result = subprocess.run(["sbatch", str(script_path)], 
    #                            capture_output=True, text=True)
    #     if result.returncode == 0:
    #         job_id = result.stdout.strip().split()[-1]
    #         print(f"Batch {i}: {script_path.name} -> Job {job_id}")
    #     else:
    #         print(f"Failed to submit batch {i}: {result.stderr}")
    
    print(f"All {len(batches)} batches submitted!")

if __name__ == "__main__":
    main()
