#!/bin/bash

export max_parallel=24   # Fix number of slurm jobs to 16

lib=${1:-"torch"}
low=${21:--1}
high=${3:--1}

if [ "$lib" == "pytorch" ]; then
    lib="torch"
elif [ "$lib" == "tensorflow" ]; then
    lib="tf"
fi

job_name=orcl
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm
export slurm_time="8:00:00"

bash $slurm_sh "python -m eval.oracle" ${job_name} ${lib} ${low} ${high}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/oracle_results_${lib}
result=$PROJECT_DIR/.tmp/oracle_result_${lib}.csv
echo "api,nominal,invalid,cpu_crash,gpu_crash,cpu_excp,gpu_excp,cpu_only_excp,gpu_only_excp,inconsistent,max_diff" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Results saved in ${result}"

source venv/bin/activate
python -m utils.aggregate_oracle_result ${result}