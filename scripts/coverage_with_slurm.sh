#!/bin/bash

export slurm_time="2:00:00" # Time limit is 2 hours
export max_parallel=690   # Fix number of slurm jobs to 690

n_inputs=${1:-500}

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

job_name=pat
echo "Patching code before running coverage script"
bash $slurm_sh "python -m eval.patching" ${job_name} ${n_inputs}

export setup_env=0       # Do not setup the environment again inside slurm script
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt
# Install instrumented pytorch
if [ ! -f $PROJECT_DIR/instrumented_pytorch/torch-* ]; then  # Download only if not already downloaded
    pip install gdown
    gdown --fuzzy https://drive.google.com/file/d/1GqydzvLO7XTlFXnSum_zhEulJpC2JRwU/view?usp=sharing -O $PROJECT_DIR/instrumented_pytorch/
fi
pip install $PROJECT_DIR/instrumented_pytorch/torch-*
export OMP_NUM_THREADS=1    # To prevent issues with coverage collection due to multithreading

job_name=cov
echo "Running coverage script"
bash $slurm_sh "python -m eval.coverage" ${job_name}

# Re-install vanilla pytorch
pip install -r $PROJECT_DIR/requirements.txt

# Aggregating and saving results: validity
valid_results=$PROJECT_DIR/.tmp/validity_results
result=$PROJECT_DIR/.tmp/validity.csv
echo "api,valid,invalid,crash,exception,total,valid_prcnt" > ${result}
for filename in ${valid_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Validity results saved in ${result}"

# Aggregating and saving results: coverage
cov_results=$PROJECT_DIR/.tmp/coverage_results
result=$PROJECT_DIR/.tmp/coverage.csv
echo "api,coverage,line_coverage,return_code" > ${result}
for filename in ${cov_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Coverage results saved in ${result}"