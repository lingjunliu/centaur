#!/bin/bash

duration=${1:-300}  # seconds

job_name=infer
slurm_sh=`dirname "$(realpath "$0")"`/slurm_base.sh # base script for slurm

bash $slurm_sh "python -m learner.invariant_inference" ${job_name} ${duration}