#!/bin/bash

n_inputs=${1:-0}      # Pass 0 to run for all inputs, otherwise, mention value
lib=${2:-tf}       # Lib: torch or tf
n_proc=${3:-100}      # Number of processes to run in parallel
method=${4:-html}     # Method to run, default is html (supports lcov too)
native=${5:-False}    # Limit the coverage to the native folder only (only applicable to the html method)
merged=${6:-False}    # To merge coverage data from multiple runs, pass True
manual_dir=${7:-""}  # To provide a manual directory for inputs, pass the path

# Preset params
save_lcov=0     # Whether to save lcov files or not
timeout=7200    # Timeout for each coverage collection process in seconds

export max_parallel=${n_proc}     # Fix number of jobs to run at a time
export elements_file=${lib}_apis.txt

# Tensorflow envrironment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1    # To prevent issues with coverage collection due to multithreading
export OMP_THREAD_LIMIT=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# alias
if [ "$lib" = "pytorch" ]; then
    lib=torch
elif [ "$lib" = "tensorflow" ]; then
    lib=tf
fi

if [ "$lib" = "torch" ]; then
    lib_v=2.2.0
    lib_ins="torch==${lib_v}"
elif [ "$lib" = "tf" ]; then
    lib_v=2.16.1
    lib_ins="tensorflow==${lib_v}"
fi

PROJECT_DIR=`dirname "$(realpath "$0")"`/..

export setup_env=0       # Do not setup the environment again inside parallel script

if [ "$lib" = "torch" ]; then
    if ! command -v python3.12 &> /dev/null; then
        echo "Error: python3.12 is not installed. Please install it before running this script."
        exit 1
    fi
    # python3.12 -m venv venv
    # source venv/bin/activate
else
    if ! command -v python3.11 &> /dev/null; then
        echo "Error: python3.11 is not installed. Please install it before running this script."
        exit 1
    fi
    # python3.11 -m venv venv311
    # source venv311/bin/activate
fi

pip install -r $PROJECT_DIR/requirements_coverage.txt

job_name=pat
echo "Patching code before running coverage script"
if [ "$merged" = "True" ] || [ "$merged" = "true" ]; then
    python -m utils.run_parallel "python -m eval.patching" "${n_inputs} ${lib} ${manual_dir}" "" "" ${job_name} ${max_parallel}
    timeout=2400
else
    python -m utils.run_parallel "python -m eval.patching" "${n_inputs} ${lib}" "" "" ${job_name} ${max_parallel}
fi

if [ "$lib" = "torch" ]; then
    # Install instrumented pytorch
    if [ ! -f ${PROJECT_DIR}/instrumented_pytorch/torch-${lib_v}* ]; then  
        echo "Error: Instrumented pytorch not found. Please copy the wheel file to ${PROJECT_DIR}/instrumented_pytorch/."
        exit 1
    fi
    pip install $PROJECT_DIR/instrumented_pytorch/torch-${lib_v}* --force-reinstall
    export OMP_NUM_THREADS=1    # To prevent issues with coverage collection due to multithreading
else
    # Install instrumented tensorflow
    if [ ! -f ${PROJECT_DIR}/instrumented_tf/tensorflow-${lib_v}* ]; then  
        echo "Error: Instrumented tensorflow not found. Please copy the wheel file to ${PROJECT_DIR}/instrumented_tf/."
        exit 1
    fi
    pip install $PROJECT_DIR/instrumented_tf/tensorflow-${lib_v}* --force-reinstall
fi

job_name=cov
echo "Running coverage script"
result=$PROJECT_DIR/.tmp/coverage_${lib}.csv
echo "api,SLATE,line_cov_SLATE" > ${result}
python -m utils.run_parallel "python -m eval.coverage" "${lib} ${method} ${native} ${save_lcov} ${timeout} ${merged}" "$PROJECT_DIR/.tmp/coverage_results" ${result} ${job_name} ${max_parallel}

if [ "$merged" = "True" ] || [ "$merged" = "true" ]; then
    python -m utils.merge_profdata .tmp/centaur_${lib}.csv ${lib}
fi

# Re-install vanilla library
pip install ${lib_ins} --force-reinstall

if [ "$merged" = "True" ] || [ "$merged" = "true" ]; then
    echo "Coverage results saved in .tmp/centaur_${lib}.csv"
else
    echo "Coverage results saved in ${result}"
fi

# Clean up temporary files
echo "Cleaning up temporary files"
rm -r $PROJECT_DIR/eval/patched_drivers
rm -r .tmp/coverage_raw_files