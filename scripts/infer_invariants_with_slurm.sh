#!/bin/bash

duration=${1:-300}  # seconds
regen=${2:-0}       # 1 means force invariant regenration
lib=${3-torch}      # library: torch or tf

# alias
if [ "$lib" = "pytorch" ]; then
  lib=torch
elif [ "$lib" = "tensorflow" ]; then
  lib=tf
fi

job_name=infer
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m learner.invariant_inference" ${job_name} ${duration} ${regen} ${lib}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/infer_results_${lib}
result=$PROJECT_DIR/.tmp/infer_result_${lib}.csv
echo "api,valid,invalid,valid_prcnt" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Results saved in ${result}"