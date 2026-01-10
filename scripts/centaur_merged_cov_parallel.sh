#!/bin/bash

dir=$1
lib=$2
n_procs=${3:-100}

subdirs=( "0-60" "60-120" "120-180" "180-240" "240-300" "300-360" "360-420" "420-480" "480-540" "540-600" )

for subdir in "${subdirs[@]}"; do
    if [ -d "${dir}/${subdir}" ]; then
        echo "Processing directory: $subdir"
        bash scripts/coverage_parallel.sh 0 $lib $n_procs html False True "${dir}/${subdir}"
    fi
done

cat .tmp/centaur_${lib}.csv
mv .tmp/merged_coverage .tmp/centaur_${lib}_profdata