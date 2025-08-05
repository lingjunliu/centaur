#!/bin/bash

# This script orchestrates the entire pipeline for DLL fuzzing with input invariants.
# From inferring invariants to fuzzing and collecting coverage.

if [ "$#" -eq 0 ]; then
  echo "Error: No arguments provided."
  echo "Usage: $0 <library> (torch or tf"
  exit 1
fi

lib=$1        # Library (torch or tf)
retry=${2:-0} # Retry flag (0 means no retry, 1 means retry cancelled jobs)
seed=200      # Seed for random number generation

# Set environment variables for Slurm
export max_parallel=160          # Maximum number of parallel jobs (set this based on the number of slurm jobs you want to spawn to run at the same time)
export max_memory_usage=90      # Maximum memory usage in percentage (set this based on the percentage of memory you do not want to exceed)

# Step 1: Infer invariants: <duration> <regen> <library>
bash scripts/infer_invariants_with_slurm.sh 1200 1 $lib
if [ "$retry" -eq 1 ]; then
  # Cancelled jobs due to memory issues are retried
  python -m utils.parse_cancelled_jobs $lib
  export elements_file=.tmp/cancelled_infs_${lib}.txt  # Set the elements file for the next steps
  bash scripts/infer_invariants_with_slurm.sh 1200 1 $lib
  export elements_file=${lib}_variations.txt  # Restore elements file for the next steps
fi
# Step 2: Generate models: <duration> <n_models> <library> <seed> <regen>
bash scripts/generate_models_with_slurm.sh 3600 0 $lib $seed 1
if [ "$retry" -eq 1 ]; then
  # Cancelled jobs due to memory issues are retried
  python -m utils.parse_cancelled_jobs $lib
  export elements_file=.tmp/cancelled_modls_${lib}.txt  # Set the elements file for the next steps
  bash scripts/generate_models_with_slurm.sh 3600 0 $lib $seed 1
  export elements_file=${lib}_apis.txt  # Restore elements file for the next steps
fi
# Step 3: Fuzz with the generated models: <duration> <n_inputs> <library> <seed>
bash scripts/fuzz_with_slurm.sh 180 0 $lib $seed
if [ "$lib" = "torch" ]; then
  # Step 4: Collect coverage: <n_inputs>
  bash scripts/coverage_with_slurm.sh 0 $lib html False
elif [ "$lib" = "tf" ]; then
  # Step 4: Collect coverage using Docker
  docker build -t tf_216_instr_im . -f instrumented_tf/Dockerfile
  docker run --name tf_216_instr tf_216_instr_im bash -c "cd /workspace/repo && bash scripts/coverage_parallel.sh 0 tf ${max_parallel} html False"
  docker cp tf_216_instr:/workspace/repo/.tmp/coverage_tf.csv .tmp/coverage_tf.csv
  docker rm -f tf_216_instr
else
  echo "Error: Unsupported library '$lib'. Supported libraries are 'torch' and 'tf'."
  exit 1
fi