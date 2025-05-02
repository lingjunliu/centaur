#!/bin/bash

export max_parallel=4   # Fix number of slurm jobs to 4

job_name=orcl
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m eval.oracle" ${job_name}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/oracle_results
result=$PROJECT_DIR/.tmp/oracle_result.csv
echo "api,nominal,invalid,cpu_crash,gpu_crash,inconsistent" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Results saved in ${result}"