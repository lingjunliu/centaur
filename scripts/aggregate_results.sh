dir=${1}
csv=${2}
header=${3}
pattern=${4:-*.csv}

# Aggregating and saving results
PROJECT_DIR=`dirname "$(realpath "$0")"`/..
tmp_results=$PROJECT_DIR/.tmp/${dir}
result=$PROJECT_DIR/.tmp/${csv}
echo ${header} > ${result}
for filename in ${tmp_results}/${pattern}
do
    cat ${filename} >> ${result}
done

echo "Results saved in ${result}"