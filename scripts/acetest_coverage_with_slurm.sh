#!/bin/bash

export slurm_time="2:00:00" # Time limit is 2 hours
export max_parallel=690   # Fix number of slurm jobs to 690

dir=$1

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

job_name=apat
echo "Patching code before running coverage script"
bash $slurm_sh "python -m eval.acetest.patching" ${job_name} ${dir}

export setup_env=0       # Do not setup the environment again inside slurm script
if ! command -v python3.12 &> /dev/null; then
    echo "Error: python3.12 is not installed. Please install it before running this script."
    exit 1
fi
python3.12 -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt
# Install instrumented pytorch
if [ ! -f $PROJECT_DIR/instrumented_pytorch/torch-* ]; then  # Download only if not already downloaded
    pip install gdown
    gdown --fuzzy https://drive.google.com/file/d/1GqydzvLO7XTlFXnSum_zhEulJpC2JRwU/view?usp=sharing -O $PROJECT_DIR/instrumented_pytorch/
fi
pip install $PROJECT_DIR/instrumented_pytorch/torch-*
export OMP_NUM_THREADS=1    # To prevent issues with coverage collection due to multithreading

job_name=acov
echo "Running coverage script"
bash $slurm_sh "python -m eval.acetest.coverage" ${job_name}

# Re-install vanilla pytorch
pip install -r $PROJECT_DIR/requirements.txt

# Aggregating and saving results: coverage
cov_results=$PROJECT_DIR/.tmp/acetest_coverage
result=$PROJECT_DIR/.tmp/acetest_coverage.csv
echo "api,coverage,line_coverage,return_code" > ${result}
for filename in ${cov_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Coverage results saved in ${result}"