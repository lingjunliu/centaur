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
max_parallel=16
mkdir -p logs
job_name=infer

for api in "${apis[@]}"; do
    sbatch -c 1 \
        --job-name=${job_name}-${i} \
        --output="logs/${api}_inv.out" \
        --wrap="srun --cpu-bind=cores python -m learner.invariant_inference ${api} ${duration}"
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