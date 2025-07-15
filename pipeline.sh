#!/bin/bash

# This script orchestrates the entire pipeline for DLL fuzzing with input invariants.
# From inferring invariants to fuzzing and collecting coverage.

if [ "$#" -eq 0 ]; then
  echo "Error: No arguments provided."
  echo "Usage: $0 <library> (torch or tf"
  exit 1
fi

lib=$1      # Library (torch or tf)
seed=200    # Seed for random number generation

# Set environment variables for Slurm
export max_parallel=100          # Maximum number of parallel jobs (set this based on the number of slurm jobs you want to spawn to run at the same time)
export max_memory_usage=90      # Maximum memory usage in percentage (set this based on the percentage of memory you do not want to exceed)

# Step 1: Infer invariants: <duration> <regen> <library>
bash scripts/infer_invariants_with_slurm.sh 1200 0 $lib
# Step 2: Generate models: <duration> <n_models> <library> <seed> <regen>
bash scripts/generate_models_with_slurm.sh 3600 0 $lib $seed 1
# Step 3: Fuzz with the generated models: <duration> <n_inputs> <library> <seed>
bash scripts/fuzz_with_slurm.sh 180 0 $lib $seed
# Step 4: Collect coverage: <n_inputs>
bash scripts/coverage_with_slurm.sh 0 $lib html False