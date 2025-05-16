#!/bin/bash
source /projects/bdfv/spack/share/spack/setup-env.sh && spack load python@3.10.14

which python

export max_parallel=349   # Fix number of slurm jobs to 349

low=${1:--1}
high=${2:--1}

job_name=orcl
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base_ncsa.sh # base script for slurm

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

source venv/bin/activate
python -m utils.aggregate_oracle_result ${result}
