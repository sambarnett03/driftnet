#!/bin/bash

# 1. Create a unique folder for this specific run based on the date/time
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_DIR="../slurm_work/job_$TIMESTAMP"
mkdir -p "$JOB_DIR"

# 2. Copy your code and the Slurm script into that folder
cp -R scripts slurm_scripts src $JOB_DIR
# 3. Move into that folder and submit the job
cd "$JOB_DIR"
sbatch slurm_scripts/run_$1.sh

echo "Job submitted from snapshot: $JOB_DIR"
