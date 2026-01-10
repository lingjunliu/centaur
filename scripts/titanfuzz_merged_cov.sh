#!/bin/bash

dir=$1
lib=$2

subdirs=( "0-60" "60-120" "120-180" "180-240" "240-300" "300-360" "360-420" "420-480" "480-540" "540-600" )

for subdir in "${subdirs[@]}"; do
    if [ -d "${dir}/${subdir}" ]; then
        echo "Processing directory: $subdir"
        bash eval/titanfuzz/cov_with_monke_titanfuzz.sh "${dir}/${subdir}" $lib 0 1 1 1 True
        mv .tmp/modified_inputs .tmp/$subdir
    fi
done

if [ $lib == "torch" ]; then
    mv .tmp/merged_coverage .tmp/titanfuzz_${lib}_profdata
    cat .tmp/titanfuzz_${lib}.csv
fi