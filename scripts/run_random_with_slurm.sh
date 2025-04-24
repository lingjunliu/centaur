#!/bin/bash

duration=${1:-300}  # seconds

job_name=rand
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m generator.random_generation" ${job_name} ${duration}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/rand_results
result=$PROJECT_DIR/.tmp/rand_result.csv
echo "api,valid,invalid,valid_prcnt" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done
rm -r ${tmp_results}

echo "Results saved in ${result}"