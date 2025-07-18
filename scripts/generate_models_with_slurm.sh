#!/bin/bash

duration=${1:-300}    # seconds
n_max=${2:-0}         # define maximum number of inputs to generate, 0 means no max
lib=${3:-torch}       # library: torch or tf
seed=${4:-200}        # random seed for the generator
regen=${5:-0}         # regenerate models if they already exist

# alias
if [ "$lib" = "pytorch" ]; then
  lib=torch
elif [ "$lib" = "tensorflow" ]; then
  lib=tf
fi

if [ -z "${elements_file}" ]; then
  export elements_file=${lib}_variations.txt
fi
export OMP_NUM_THREADS=1    # To prevent issues with coverage collection due to multithreading
export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN optimizations for TensorFlow

# Add 10 minutes (600 seconds) to account for rule refinement
total_seconds=$((duration + 600))

# Convert to HH:MM:SS
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))
seconds=$((total_seconds % 60))
export slurm_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

job_name=modl
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m generator.z3" ${job_name} ${duration} ${n_max} ${lib} ${seed} ${regen}

PROJECT_DIR=`dirname "$(realpath "$0")"`/..

# Aggregating and saving results
tmp_results=$PROJECT_DIR/.tmp/model_results
result=$PROJECT_DIR/.tmp/model_generation_$lib.csv
echo "api,unsat,nominal,invalid,crash,exception,total,valid_prcnt" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Model gen results saved in ${result}"