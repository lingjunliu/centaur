#!/bin/bash

duration=${1:-300}    # seconds
n_max=${2:-0}         # define maximum number of inputs to generate, 0 means no max
lib=${3:-torch}       # library: torch or tf
seed=${4:-200}        # random seed for the generator
print_details=${5:-False} # print additional details
max_parallel=${6:-100}  # number of parallel processes

PROJECT_DIR=`dirname "$(realpath "$0")"`/..

# alias
if [ "$lib" = "pytorch" ]; then
  lib=torch
elif [ "$lib" = "tensorflow" ]; then
  lib=tf
fi

export elements_file=${lib}_apis.txt

job_name=dllf
result=$PROJECT_DIR/.tmp/fuzz_result_$lib.csv
echo "api,n_models,nominal,invalid,crash,exception,total,valid_prcnt" > ${result}

python -m utils.run_parallel "python -m generator.fuzz" "${duration} ${n_max} ${lib} ${seed} ${print_details}" "$PROJECT_DIR/.tmp/fuzz_results" ${result} ${job_name} ${max_parallel}

echo "Fuzzing results saved in ${result}"

crash_results=$PROJECT_DIR/.tmp/crash_logs
crash_log=$PROJECT_DIR/.tmp/crashes.log
cat ${crash_results}/*.log > ${crash_log}

echo "Crashes saved in ${crash_log}"