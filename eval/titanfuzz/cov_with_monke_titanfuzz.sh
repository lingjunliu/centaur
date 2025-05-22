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
        ((n++))
        printf "${remaining} jobs remaining, ${running} running, ${pending} waiting| ${n} seconds elapsed\r"
    done
}

##################################################################################

MAX_INPUTS=${1:-500}
APPLY_MONKE=${2:-1}
RUN_MOD=${3:-1}
COMPUTE_COV=${4:-1}

conda_env_name=torch310 # conda environment name with instrumented torch
DIR=${root_dir}/eval/titanfuzz/titanfuzz_inputs/Results_690_apis/torch/valid
out_dir=${root_dir}/eval/titanfuzz/modified_inputs

apisFile=${root_dir}/apis.txt
declare -a apis
apis=(`cat "$apisFile"`)

n_procs=192
time_interval=10

# Clearing old results
if [ ${APPLY_MONKE} -eq 1 ]; then
    rm -r $out_dir > /dev/null 2>&1
elif [ ${RUN_MOD} -eq 1 ]; then
    rm -r $out_dir/*/ > /dev/null 2>&1
fi

outputs=$root_dir/.tmp/titanfuzz_results
logs=${outputs}/logs

mkdir -p $out_dir
mkdir -p ${logs}
mkdir -p ${outputs}
result_file=${outputs}/coverage.csv

source ${root_dir}/venv/bin/activate

total_files=$(find ${DIR} -name "*.py" -type f | wc -l)
inputs_per_proc=$(((total_files + n_procs - 1) / n_procs))

if [ ${APPLY_MONKE} -eq 1 ]; then
    find ${DIR} -name "*.py" -type f > list_of_input_files
    split -l ${inputs_per_proc} --numeric-suffixes list_of_input_files input_files_

    for input_file in input_files_*; do
        [ -e $input_file ] || continue # ignoring the pattern itself
        log_file=${logs}/monke_${input_file}.log
        err_file=${logs}/monke_${input_file}.error

        sbatch -c 1 --mem-per-cpu 1G -t 2:00:00 -J $sota --wrap "bash ${root_dir}/eval/titanfuzz/batch_monkey_patching.sh ${input_file} ${out_dir}" --output ${log_file} --error ${err_file}
    done

    wait_for_slurm ${time_interval} ${sota} "monkey patching"

    # clean up
    rm input_files_*
fi

if [ ${RUN_MOD} -eq 1 ]; then
    find . -name "*.py" -type f > list_of_input_files
    split -l ${inputs_per_proc} --numeric-suffixes list_of_input_files input_files_

    for input_file in input_files_*; do
        [ -e $input_file ] || continue # ignoring the pattern itself
        log_file=${logs}/mod_${input_file}.log
        err_file=${logs}/mod_${input_file}.error

        sbatch -c 1 --mem-per-cpu 1G -t 2:00:00 -J $sota --wrap "bash ${root_dir}/eval/titanfuzz/batch_input_running.sh ${input_file}" --output ${log_file} --error ${err_file}
    done

    wait_for_slurm ${time_interval} ${sota} "run modified code"
    # clean up
    rm input_files_*
fi

# Activate conda environment for coverage computation
deactivate
conda init > /dev/null 2>&1
eval "$(conda shell.bash hook)" > /dev/null 2>&1

if [ ${COMPUTE_COV} -eq 1 ]; then
    conda activate torch310
    libname=torch
    export TORCH_BUILD_DIR=$(pip show "$libname" | grep "Location:" | awk '{print $2}')/${libname}
    echo "Using ${libname} from ${TORCH_BUILD_DIR}"
    echo $PWD
    for api in "${apis[@]}"
    do
        log_file=${logs}/cov_${api}.log
        err_file=${logs}/cov_${api}.error
        out_file=${outputs}/${api}.txt

        sbatch -c 1 --mem-per-cpu 1G -t 2:00:00 -J $sota --wrap "python -m eval.titanfuzz.compute_coverage_titanfuzz ${out_dir} ${api} ${out_file} ${MAX_INPUTS}" --output ${log_file} --error ${err_file}
    done

    wait_for_slurm ${time_interval} ${sota} "computing coverage"

    rm ${result_file}
    printf "api,coverage,line_coverage,n_inputs\n" >> ${result_file}
    for api in "${apis[@]}"
    do
        api_out=${outputs}/${api}.txt
        if [ -f $api_out ]; then
            cat $api_out >> ${result_file}
            rm $api_out
        fi
    done

    conda deactivate
fi

echo "Results are saved in ${result_file}"

# Aggregating and saving results: validity
result=${outputs}/validity.csv
echo "api,valid,invalid,crash,exception,total,valid_prcnt" > ${result}
for filename in ${out_dir}/*/*.csv
do
    cat ${filename} >> ${result}
done

echo "Validity results saved in ${result}"