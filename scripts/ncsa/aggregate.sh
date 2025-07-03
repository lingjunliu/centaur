# Aggregating and saving results
spack load py-pip@23.1.2 ^python@3.12.5

lib=torch

PROJECT_DIR=`dirname "$(realpath "$0")"`/../..
tmp_results=$PROJECT_DIR/.tmp/oracle_results_${lib}
result=$PROJECT_DIR/.tmp/oracle_result_${lib}.csv
echo "api,nominal,invalid,cpu_crash,gpu_crash,cpu_excp,gpu_excp,cpu_only_excp,gpu_only_excp,inconsistent,max_diff" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done

echo "Results saved in ${result}"

source venv/bin/activate
python -m utils.aggregate_oracle_result ${result}
