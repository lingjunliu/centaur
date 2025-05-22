#!/bin/bash

sota=tfuz

root_dir=$(git rev-parse --show-toplevel)
export PYTHONPATH=${PYTHONPATH}:${root_dir}

##################################################################################

# Function: Wait for slurm jobs to finish
wait_for_slurm(){
    n=0
    wait_time=$1
    job_name=$2
    job_type=$3
    while [ ! $(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l) -eq "0" ];
    do
        sleep 1s
        remaining=$(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l)
        pending=$(squeue --user=$USER --state=PENDING | grep -vE "JOBID" | grep "${job_name}" | wc -l)
        running=$(squeue --user=$USER --state=RUNNING | grep -vE "JOBID" | grep "${job_name}" | wc -l)
        if (( n % wait_time == 0 )); then
            python ${root_dir}/src/send_status_update.py "${remaining} ${job_type} jobs remaining in the slurm queue | ${running} running, ${pending} waiting to run | ${n} seconds elapsed"
        fi
        ((n++))
        printf "${remaining} jobs remaining, ${running} running, ${pending} waiting| ${n} seconds elapsed\r"
    done
}

##################################################################################

DIR=${1:-titanfuzz_inputs/apis_189/torch/valid}
MAX_INPUTS=${2:-500}
APPLY_MONKE=${3:-1}
RUN_MOD=${4:-1}
COMPUTE_COV=${5:-1}
out_dir=${6:-modified_inputs}

conda init > /dev/null 2>&1
eval "$(conda shell.bash hook)" > /dev/null 2>&1

apisFile=${root_dir}/src/input_apis
declare -a apis
apis=(`cat "$apisFile"`)

n_procs=200
time_interval=300

if [ ${APPLY_MONKE} -eq 1 ]; then
    rm -r $out_dir > /dev/null 2>&1
elif [ ${RUN_MOD} -eq 1 ]; then
    rm -r $out_dir/*/ > /dev/null 2>&1
fi

mkdir -p $out_dir
mkdir -p logs
mkdir -p outputs
result_file=outputs/coverage.csv

conda activate torch_nightly

total_files=$(find ${DIR} -name "*.py" -type f | wc -l)
inputs_per_proc=$(((total_files + n_procs - 1) / n_procs))

if [ ${APPLY_MONKE} -eq 1 ]; then
    find ${DIR} -name "*.py" -type f > list_of_input_files
    split -l ${inputs_per_proc} --numeric-suffixes list_of_input_files input_files_

    for input_file in input_files_*; do
        [ -e $input_file ] || continue # ignoring the pattern itself
        log_file=logs/${input_file}.log
        err_file=logs/${input_file}.error

        sbatch -c 1 --mem-per-cpu 1G -t 2:00:00 -J $sota --wrap "bash batch_monkey_patching.sh ${input_file} ${out_dir}" --output ${log_file} --error ${err_file}
    done

    wait_for_slurm ${time_interval} ${sota} "monkey patching"

    # clean up
    rm input_files_*
fi

if [ ${RUN_MOD} -eq 1 ]; then
    cp batch_input_running.sh ${out_dir}/
    cd ${out_dir}
    mkdir -p logs

    find . -name "*.py" -type f > list_of_input_files
    split -l ${inputs_per_proc} --numeric-suffixes list_of_input_files input_files_

    for input_file in input_files_*; do
        [ -e $input_file ] || continue # ignoring the pattern itself
        log_file=logs/${input_file}.log
        err_file=logs/${input_file}.error

        sbatch -c 1 --mem-per-cpu 1G -t 2:00:00 -J $sota --wrap "bash batch_input_running.sh ${input_file}" --output ${log_file} --error ${err_file}
    done

    wait_for_slurm ${time_interval} ${sota} "run modified code"
    # clean up
    rm input_files_*

    cd ${root_dir}/src/titanfuzz_utils
fi

conda deactivate

if [ ${COMPUTE_COV} -eq 1 ]; then
    conda activate torch310
    libname=torch
    export TORCH_BUILD_DIR=$(pip show "$libname" | grep "Location:" | awk '{print $2}')/${libname}
    echo "Using ${libname} from ${TORCH_BUILD_DIR}"
    echo $PWD
    for api in "${apis[@]}"
    do
        log_file=logs/${api}.log
        err_file=logs/${api}.error
        out_file=outputs/${api}.txt

        sbatch -c 1 --mem-per-cpu 1G -t 2:00:00 -J $sota --wrap "python compute_coverage_titanfuzz.py ${out_dir} ${api} ${out_file} ${MAX_INPUTS}" --output ${log_file} --error ${err_file}
    done

    wait_for_slurm ${time_interval} ${sota} "computing coverage"

    rm ${result_file}
    printf "api,coverage,line_coverage,n_inputs\n" >> ${result_file}
    for api in "${apis[@]}"
    do
        api_out=outputs/${api}.txt
        if [ -f $api_out ]; then
            cat $api_out >> ${result_file}
            rm $api_out
        fi
    done

    conda deactivate
fi

echo "Results are saved in ${result_file}"