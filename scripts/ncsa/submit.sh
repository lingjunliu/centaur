#!/bin/bash
# Submit all batch scripts in the current directory

for script in batch_*.sh; do
    sbatch "$script"
done
