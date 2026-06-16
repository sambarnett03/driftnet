#!/bin/bash

# Ensure exactly one argument is passed
if [ "$#" -ne 1 ]; then
    echo "Error: Exactly one argument required." >&2
    echo "Usage: $0 [orchid|lotus]" >&2
    exit 1
fi

# Validate that the argument is either 'orchid' or 'lotus'
case "$1" in
    orchid|lotus)
        # Argument is valid, proceed with the script
        ;;
    *)
        echo "Error: Invalid argument '$1'. Must be 'orchid' or 'lotus'." >&2
        echo "Usage: $0 [orchid|lotus]" >&2
        exit 1
        ;;
esac

# 1. Create a unique folder for this specific run based on the date/time
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_DIR="../slurm_work/job_$TIMESTAMP"
mkdir -p "$JOB_DIR"

# 2. Copy your code and the Slurm script into that folder
cp -R scripts slurm_scripts src "$JOB_DIR"

# 3. Move into that folder and submit the job
cd "$JOB_DIR" || exit 1
sbatch slurm_scripts/run_"$1".sh

echo "Job submitted from snapshot: $JOB_DIR"