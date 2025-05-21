#!/bin/bash

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export PYTHONWARNINGS="ignore"
# Tensorflow envrironment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

lib=${1:-"torch"}
low=${21:--1}
high=${3:--1}

if [ "$lib" == "pytorch" ]; then
    lib="torch"
elif [ "$lib" == "tensorflow" ]; then
    lib="tf"
fi

# Creating virtual environment
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt

cd $PROJECT_DIR

apis=(`cat apis.txt`)
n_apis=${#apis[@]}
i=0

# Running oracle
for api in "${apis[@]}"; do
    ((i++))
    python -m eval.crash_monitor ${api} ${lib} ${low} ${high}
    echo "Finished ${i}/${n_apis}"
done

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