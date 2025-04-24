#!/bin/bash

duration=${1:-300}  # seconds

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export PYTHONWARNINGS="ignore"

source ${PROJECT_DIR}/scripts/utils.sh

# Creating virtual environment
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt

# Running random generation
cd $PROJECT_DIR

apis=(`cat apis.txt`)
n_apis=${#apis[@]}
i=0
elapsed=0
max_parallel=60
mkdir -p logs
job_name=dllf

for api in "${apis[@]}"; do
    sbatch -c 1 \
        --job-name=${job_name}-${i} \
        --output="logs/${api}.out" \
        --wrap="srun --cpu-bind=cores python -m generator.fuzz ${api} ${duration}"
    ((i++))

    # limit number of running jobs
    while (( $(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l) >= max_parallel )); do
        print_progress ${job_name} ${elapsed} "${i}/${n_apis}"
        sleep 10
        (( elapsed = elapsed + 10 ))
    done
done

# wait for everything to finish
while (( $(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l) > 0 )); do
    print_progress ${job_name} ${elapsed} "${i}/${n_apis}"
    sleep 10
    (( elapsed = elapsed + 10 ))
done

# Aggregating and saving results
tmp_results=$PROJECT_DIR/.tmp/fuzz_results
result=$PROJECT_DIR/.tmp/fuzz_result.csv
echo "api,valid,invalid,valid_prcnt" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done