#!/bin/bash

input_files=${1:-list_of_input_files}
out_dir=${2:-tmp}

root_dir=$(git rev-parse --show-toplevel)
export PYTHONPATH=${PYTHONPATH}:${root_dir}

apisFile=${root_dir}/src/input_apis

declare -a inputs
inputs=(`cat "$input_files"`)

echo "Starting monkey patching"

n=0
for input_file in "${inputs[@]}"; do
    python -m eval.titanfuzz.monkey_patching ${input_file} ${apisFile} ${out_dir}
    if ((n % 100 == 0)); then
        echo "${n}/${#inputs[@]} done"
    fi
    ((n++))
done

echo "Done monkey patching"