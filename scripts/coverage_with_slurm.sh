#!/bin/bash

export slurm_time="0:10:00" # Time limit is 10 minutes
export max_parallel=690   # Fix number of slurm jobs to 690
export setup_env=0       # Do not setup the environment again, run this script with conda environment

n_inputs=${1:-500}

slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

job_name=pat
echo "Patching code before running coverage script"
bash $slurm_sh "python -m eval.patching" ${job_name} ${n_inputs}

job_name=cov
echo "Running coverage script"
bash $slurm_sh "python -m eval.coverage" ${job_name}

# Aggregating and saving results: validity
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
valid_results=$PROJECT_DIR/.tmp/validity_results
result=$PROJECT_DIR/.tmp/validity.csv
echo "api,valid,invalid,crash,exception,total,valid_prcnt" > ${result}
for filename in ${valid_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Validity results saved in ${result}"

# Aggregating and saving results: coverage
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
cov_results=$PROJECT_DIR/.tmp/coverage_results
result=$PROJECT_DIR/.tmp/coverage.csv
echo "api,coverage,line_coverage,return_code" > ${result}
for filename in ${cov_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Coverage results saved in ${result}"