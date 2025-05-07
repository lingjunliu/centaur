#!/bin/bash

duration=${1:-300}  # seconds
mode=${2:-z3}   # z3 or optimizer
n_max=${3:-0}   # define maximum number of inputs to generate, 0 means no max
limit=${4:-30}  # optimizer will random restart after <limit> seconds

job_name=dllf
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m generator.fuzz" ${job_name} ${duration} ${mode} ${n_max} ${limit}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/fuzz_results
result=$PROJECT_DIR/.tmp/fuzz_result.csv
echo "api,valid,invalid,crash,total,valid_prcnt" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done
rm -r ${tmp_results}

echo "Results saved in ${result}"