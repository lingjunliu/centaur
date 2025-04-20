############################################################################

# Function: Wait for slurm jobs to finish
wait_for_slurm(){
    job_name=$1
    n=0
    wait_time=10    # seconds
    while [ ! $(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l) -eq "0" ];
    do        
        sleep 1s        
        remaining=$(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l)
        pending=$(squeue --user=$USER --state=PENDING | grep -vE "JOBID" | grep "${job_name}" | wc -l)
        running=$(squeue --user=$USER --state=RUNNING | grep -vE "JOBID" | grep "${job_name}" | wc -l)        
        if (( n % wait_time == 0 )); then
            echo "${remaining}  jobs remaining in the slurm queue | ${running} running, ${pending} waiting to run | ${n} seconds elapsed"
        fi
        ((n++))
    done
}

############################################################################

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Creating virtual environment
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt

# Running random generation
cd $PROJECT_DIR
apis=(`cat apis.txt`)
job_name="rand_gen"
for api in "${apis[@]}"
do
    sbatch -c 1 --mem-per-cpu 1G -t 10:00:00 -J $job_name --wrap "python -m generator.random_generation ${api}"
done
wait_for_slurm $job_name

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