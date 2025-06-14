#!/bin/bash

# Function: Print progress
print_progress(){
    job_name=$1
    elapsed=$2
    spawned=$3
    remaining=$(squeue --user=$USER | grep -vE "JOBID" | grep "${job_name}" | wc -l)
    pending=$(squeue --user=$USER --state=PENDING | grep -vE "JOBID" | grep "${job_name}" | wc -l)
    running=$(squeue --user=$USER --state=RUNNING | grep -vE "JOBID" | grep "${job_name}" | wc -l)
    printf "${remaining} jobs remaining in the slurm queue | ${running} running, ${pending} waiting to run | ${elapsed} seconds elapsed | ${spawned} jobs spawned             \r"
}