#!/bin/bash

# This script can be called to run any python script
# on all elements in elements_file (default: apis.txt). 
# The condition is the first argument of the python 
# function has to be the element and the rest of the 
# arguments has to be fixed for each execution

if [ -z "${setup_env}" ]; then
    setup_env=1    # Flag to setup the environment
fi

if [ -z "${max_parallel}" ]; then
    max_parallel=673    # Fix number of jobs to run at a time if not set
fi

if [ -z "${max_memory_usage}" ]; then
    max_memory_usage=90    # Maximum system memory usage in percent
fi

if [ -z "${elements_file}" ]; then
    elements_file=apis.txt      # File containing the list of elements to loop through (default: apis.txt)
fi

echo "Using a max parallel of $max_parallel"

cmd=$1              # commmand to run parallelly
job_name=$2         # job name (for logs)

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export PYTHONWARNINGS="ignore"
# Tensorflow envrironment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

if [ $setup_env -eq 1 ]; then
    # Creating virtual environment
    if ! command -v python3.12 &> /dev/null; then
        echo "Error: python3.12 is not installed. Please install it before running this script."
        exit 1
    fi
    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r $PROJECT_DIR/requirements.txt
fi

# Running random generation
cd $PROJECT_DIR

elements=(`cat ${elements_file}`)
n_elements=${#elements[@]}
i=0
elapsed=0
mkdir -p logs

function current_jobs() {
    jobs -pr | wc -l
}

for element in "${elements[@]}"; do
    ((i++))
    wrap_cmd="${cmd} ${element} ${@:3}"
    # Run the command in the background, redirect output to log
    bash -c "${wrap_cmd}" > "logs/${element}_${job_name}.out" 2>&1 &

    # limit number of running jobs
    while (( $(current_jobs) >= max_parallel )); do
        running=$(current_jobs)
        pending=$((n_elements - i))
        total=$n_elements
        echo -ne "Progress: Running jobs: $running | Pending: $pending | Total: $total\r"
        sleep 1
        (( elapsed = elapsed + 1 ))
    done

done

# wait for everything to finish
while (( $(current_jobs) > 0 )); do
    running=$(current_jobs)
    pending=0
    total=$n_elements
    echo -ne "Progress: Running jobs: $running | Pending: $pending | Total: $total\r"
    sleep 1
    (( elapsed = elapsed + 1 ))
done
wait
echo -e "\nAll jobs completed."