#!/bin/bash

duration=${1:-300}    # seconds
n_max=${2:-0}         # define maximum number of inputs to generate, 0 means no max
lib=${3:-torch}       # library: torch or tf
seed=${4:-200}        # random seed for the generator

# alias
if [ "$lib" = "pytorch" ]; then
  lib=torch
elif [ "$lib" = "tensorflow" ]; then
  lib=tf
fi

export elements_file=${lib}_apis.txt

# Add 2 minutes (120 seconds)
total_seconds=$((duration + 120))

# Convert to HH:MM:SS
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))
seconds=$((total_seconds % 60))
export slurm_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

job_name=dllf
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m generator.fuzz" ${job_name} ${duration} ${n_max} ${lib} ${seed}

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
# Aggregating and saving results
tmp_results=$PROJECT_DIR/.tmp/fuzz_results
result=$PROJECT_DIR/.tmp/fuzz_result_$lib.csv
echo "api,n_models,nominal,invalid,crash,exception,total,valid_prcnt" > ${result}
for filename in ${tmp_results}/*${lib}.csv
do
    cat ${filename} >> ${result}
done

echo "Fuzzing results saved in ${result}"

crash_results=$PROJECT_DIR/.tmp/crash_logs
crash_log=$PROJECT_DIR/.tmp/crashes.log
for filename in ${crash_results}/*.log
do
    cat ${filename} >> ${crash_log}
done

echo "Crashes saved in ${crash_log}"