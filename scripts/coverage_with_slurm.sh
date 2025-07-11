#!/bin/bash

export slurm_time="2:00:00" # Time limit is 2 hours
export max_parallel=690     # Fix number of slurm jobs to 690

n_inputs=${1:--1}     # Pass -1 to run for all inputs, otherwise, mention value
method=${2:-html}     # Method to run, default is html (supports lcov too)
native=${3:-False}    # Limit the coverage to the native folder only (only applicable to the html method)
lib_v=${4:-2.2.0}     # Library version to use
debug=${5:-0}         # To debug coverage difference with titanfuzz, pass 1

# Only torch is supported for coverage for now
# TODO: Add support for tensorflow
lib=torch
export elements_file=${lib}_apis.txt

if [ $debug -eq 1 ]; then
    echo "Running in debugging mode"
fi

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

export setup_env=0       # Do not setup the environment again inside slurm script

python3.12 -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt
# Installing specified version of the library (torch)
pip install torch==${lib_v}
job_name=pat
echo "Patching code before running coverage script"
bash $slurm_sh "python -m eval.patching" ${job_name} ${n_inputs}

if ! command -v python3.12 &> /dev/null; then
    echo "Error: python3.12 is not installed. Please install it before running this script."
    exit 1
fi
# Install instrumented pytorch
if [ ! -f ${PROJECT_DIR}/instrumented_pytorch/torch-${lib_v}* ]; then  # Download only if not already downloaded
    pip install gdown
    gdown --fuzzy https://drive.google.com/file/d/1GqydzvLO7XTlFXnSum_zhEulJpC2JRwU/view?usp=sharing -O $PROJECT_DIR/instrumented_pytorch/
fi
pip install $PROJECT_DIR/instrumented_pytorch/torch-${lib_v}*
export OMP_NUM_THREADS=1    # To prevent issues with coverage collection due to multithreading

job_name=cov
echo "Running coverage script"
bash $slurm_sh "python -m eval.coverage" ${job_name} ${method} ${native} ${debug}

# DEBUG ################################

if [ $debug -eq 1 ]; then
    echo "Running debugging scripts"
    job_name=deb
    bash $slurm_sh "python -m debugging.compare_coverage" ${job_name}
fi

# END DEBUG ############################

# Re-install vanilla pytorch
pip install torch==${lib_v}

# DEBUG ################################

if [ $debug -eq 1 ]; then
    echo "Extracting abstracts from debugged data"
    job_name=abs
    bash $slurm_sh "python -m debugging.get_abstracts" ${job_name}
    # Aggregating debug details
    stat_results=$PROJECT_DIR/.tmp/debug_coverage
    result=$PROJECT_DIR/.tmp/debug_stats.csv
    echo "api,missing_params,different_dtype,length_mismatches,value_mismatches,ndim_mismatches,dimsize_mismatches,range_mismatches" > ${result}
    for filename in ${stat_results}/*_stats.csv
    do
        cat ${filename} >> ${result}
    done
fi

# END DEBUG ############################

# Aggregating and saving results: coverage
cov_results=$PROJECT_DIR/.tmp/coverage_results
result=$PROJECT_DIR/.tmp/coverage.csv
echo "api,SLATE,line_cov_SLATE" > ${result}
for filename in ${cov_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Coverage results saved in ${result}"

# Clean up temporary files
echo "Cleaning up temporary files"
rm -r $PROJECT_DIR/eval/patched_drivers
if [ $debug -eq 0 ]; then
    rm -r .tmp/coverage_raw_files
fi