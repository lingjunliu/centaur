#!/bin/bash

duration=${1:-300}  # seconds
regen=${2:-0}       # 1 means force invariant regenration

job_name=infer
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m learner.invariant_inference" ${job_name} ${duration} ${regen}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/infer_results
result=$PROJECT_DIR/.tmp/infer_result.csv
echo "api,valid,invalid,valid_prcnt" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Results saved in ${result}"