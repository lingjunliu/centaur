#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --output=out.log

srun --cpu-bind=cores bash ${1}