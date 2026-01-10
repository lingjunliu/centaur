 #!/bin/bash

dir=$1                  # Directory containing modified inputs each API
lib=${2:-tf}            # Lib: torch or tf
n_proc=${3:-100}        # Number of parallel processes
merged=${4:-False}      # Whether to merge all api coverage (True) or keep them separate (False)

export max_parallel=${n_proc}     # Fix number of jobs to run at a time
export elements_file=${lib}_apis.txt

if [ "$merged" = "True" ] || [ "$merged" = "true" ]; then
    timeout=2400
else
    timeout=7200
fi

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
mkdir -p $PROJECT_DIR/.tmp/titanfuzz_coverage
result=$PROJECT_DIR/.tmp/titanfuzz_coverage_${lib}.csv
# {api},{num_branches},{num_lines},{n_inputs},{return_code}
printf "api,titanfuzz,line_cov_titanfuzz,n_inputs,return_code\n" > ${result}
# python -m eval.titanfuzz.compute_coverage_titanfuzz ${api} ${out_dir} ${lib} ${out_file} ${MAX_INPUTS}
python -m utils.run_parallel "python -m eval.titanfuzz.compute_coverage_titanfuzz" "${dir} ${lib}" "$PROJECT_DIR/.tmp/titanfuzz_coverage 0 ${merged} ${timeout}" ${result} ${job_name} ${max_parallel}

if [ "$merged" = "True" ] || [ "$merged" = "true" ]; then
    python -m utils.merge_profdata .tmp/titanfuzz_${lib}.csv ${lib}
fi

# Re-install vanilla library
pip install ${lib_ins} --force-reinstall

if [ "$merged" = "True" ] || [ "$merged" = "true" ]; then
    echo "Coverage results saved in .tmp/titanfuzz_${lib}.csv"
else
    echo "Coverage results saved in ${result}"
fi

# Clean up temporary files
echo "Cleaning up temporary files"
rm -r .tmp/coverage_raw_files