#!/bin/bash

input_files=${1:-list_of_input_files}

declare -a inputs
inputs=(`cat "$input_files"`)

echo "Starting running patched python files"

n=0
for input_file in "${inputs[@]}"; do
    python ${input_file}
    if ((n % 100 == 0)); then
        echo "Running inputs | ${n}/${#inputs[@]} done"
    fi
    ((n++))
done

echo "Done running patched python files"