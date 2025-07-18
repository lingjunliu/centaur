#!/bin/bash

lib=${1:-torch}  # library: torch or tf
duration=${2:-300}  # seconds
n_max=${3:-0}   # define maximum number of inputs to generate, 0 means no max

export elements_file=${lib}_apis.txt

# Add 2 minutes (120 seconds)
total_seconds=$((duration + 120))

# Convert to HH:MM:SS
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))
seconds=$((total_seconds % 60))
export slurm_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

job_name=rand
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m generator.random_generation" ${job_name} ${duration} ${n_max} ${lib}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/rand_results_${lib}
result=$PROJECT_DIR/.tmp/rand_result_${lib}.csv
echo "api,valid,invalid,crash,total,valid_prcnt" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Results saved in ${result}"