#!/bin/bash

# This script can be called to run any python script
# on all apis in apis.txt. The condition is the first
# argument of the python function has to be the api
# and the rest of the arguments has to be fixed for
# each execution

if [ -z "${setup_env}" ]; then
    setup_env=1    # Flag to setup the environment
fi

if [ -z "${max_parallel}" ]; then
    max_parallel=690    # Fix number of slurm jobs to run at a time if not set
fi

if [ -z "${slurm_time}" ]; then
    slurm_time="2:00:00"    # Default slurm timeout
fi

if [ -z "${apis_file}" ]; then
    apis_file=apis.txt      # File containing the list of APIs
fi

echo "Using a slurm timeout of $slurm_time"

cmd=$1              # commmand to run parallelly
job_name=$2         # slurm job name

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export PYTHONWARNINGS="ignore"
# Tensorflow envrironment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

source ${PROJECT_DIR}/scripts/utils.sh

if [ $setup_env -eq 1 ]; then
    # Creating virtual environment
    python -m venv venv
    source venv/bin/activate
    pip install -r $PROJECT_DIR/requirements.txt
fi

# Running random generation
cd $PROJECT_DIR

apis=(`cat ${apis_file}`)
n_apis=${#apis[@]}
i=0
elapsed=0
mkdir -p logs

for api in "${apis[@]}"; do
    ((i++))
    wrap_cmd="${cmd} ${api} ${@:3}"
    # Run sbatch with a timeout of 2 hour
    sbatch -c 1 \
        --job-name=${job_name}-${i} \
        --output="logs/${api}_${job_name}.out" \
        --time=$slurm_time \
        --wrap="${wrap_cmd}"

    # limit number of running jobs
    while (( $(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l) >= max_parallel )); do
        print_progress ${job_name} ${elapsed} "${i}/${n_apis}"
        sleep 1
        (( elapsed = elapsed + 1 ))
    done
done

# wait for everything to finish
while (( $(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l) > 0 )); do
    print_progress ${job_name} ${elapsed} "${i}/${n_apis}"
    sleep 1
    (( elapsed = elapsed + 1 ))
done