#!/bin/bash

duration=${1:-300}  # seconds

############################################################################

# Function: Print progress
print_progress(){
    job_name=$1
    elapsed=$2
    remaining=$(squeue --user=$USER | grep -c $job_name | wc -l)
    pending=$(squeue --user=$USER --state=PENDING | grep -c $job_name | wc -l)
    running=$(squeue --user=$USER --state=RUNNING | grep -c $job_name | wc -l)
    echo "${remaining} jobs remaining in the slurm queue | ${running} running, ${pending} waiting to run | ${elapsed} seconds elapsed"
}

############################################################################

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Creating virtual environment
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt

# Running random generation
cd $PROJECT_DIR

i=0
elapsed=0
max_parallel=60
mkdir -p logs
job_name=fuzz_api-

for api in $(cat apis.txt); do
    sbatch -c 1 \
        --job-name="${job_name}${i}" \
        --output="logs/${api}.out" \
        --wrap="srun --cpu-bind=cores PYTHONWARNINGS='ignore' python -m generator.fuzz ${api} ${duration}"
    ((i++))

    # limit number of running jobs
    while (( $(squeue -u $USER | grep -c ${job_name}) >= max_parallel )); do
        sleep 10
        (( elapsed = elapsed + 10 ))
        print_progress ${job_name} ${elapsed}
    done
done

# wait for everything to finish
while (( $(squeue -u $USER | grep -c 'fuzz_api-') >= 0 )); do
    sleep 10
    (( elapsed = elapsed + 10 ))
    print_progress ${job_name} ${elapsed}
done