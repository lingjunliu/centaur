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

if [ -z "${elements_file}" ]; then
  export elements_file=${lib}_variations.txt
fi
export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN optimizations for TensorFlow

job_name=inf

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/infer_results_${lib}
result=$PROJECT_DIR/.tmp/infer_result_${lib}.csv
echo "api,valid,invalid,valid_prcnt" > ${result}

python -m utils.run_parallel "python -m learner.invariant_inference" "${duration} ${regen} ${lib}" ${tmp_results} ${result} ${job_name} ${max_parallel}

echo "Results saved in ${result}"