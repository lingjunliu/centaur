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

# TODO: Implement for tensorflow, currently only pytorch is supported
if [ "$lib" != "torch" ]; then
    echo "Currently only torch is supported, please use 'torch' as the library argument."
    exit 1
fi

export apis_file=${lib}_variations.txt
export OMP_NUM_THREADS=1    # To prevent issues with coverage collection due to multithreading

# Add 2 minutes (120 seconds)
total_seconds=$((duration + 120))

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
rm -r ${tmp_results}

echo "Model gen results saved in ${result}"