#!/bin/bash

export max_parallel=16   # Fix number of slurm jobs to 16

low=${1:--1}
high=${2:--1}

job_name=orcl
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm
export slurm_time="8:00:00"

bash $slurm_sh "python -m eval.oracle" ${job_name} ${low} ${high}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/oracle_results
result=$PROJECT_DIR/.tmp/oracle_result.csv
echo "api,nominal,invalid,cpu_crash,gpu_crash,cpu_excp,gpu_excp,cpu_only_excp,gpu_only_excp,inconsistent,max_diff" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Results saved in ${result}"

python -m utils.aggregate_oracle_result ${result}