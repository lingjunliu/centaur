PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Creating virtual environment
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt

# Running random generation
cd $PROJECT_DIR
apis=(`cat apis.txt`)
for api in "${apis[@]}"
do
    python -m generator.random_generation ${api}
done

# Aggregating and saving results
tmp_results=$PROJECT_DIR/.tmp/results
result=$PROJECT_DIR/.tmp/result.csv
echo "api,valid,invalid,valid_prcnt" > ${result}
for filename in ${tmp_results}/*.csv
do
    cat ${filename} >> ${result}
done
rm -r ${tmp_results}

echo "Results saved in ${result}"