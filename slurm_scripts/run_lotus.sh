#!/bin/bash
#SBATCH --job-name=lotus_downscale
#SBATCH --account=oxford_es
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o /home/users/sbarnett/documents/driftnet/slurm_scripts/logs/output.%j.out # STDOUT
#SBATCH -e /home/users/sbarnett/documents/driftnet/slurm_scripts/logs/output.%j.err
#SBATCH --time=01:00:00              # Time limit (hh:mm:ss)
#SBATCH --mem=128GB                    # Memory request


# 1. Set up the environment
module load jaspy/3.12/v20250704
source ~/my_venvs/driftnet_venv/bin/activate

# 2. Navigate to the directory containing your script
cd ~/documents/driftnet/

export LD_LIBRARY_PATH=/apps/jasmin/jaspy/miniforge_envs/jaspy3.12/mf3-25.3.0-3/envs/jaspy3.12-mf3-25.3.0-3-v20250704/lib:$LD_LIBRARY_PATH

python -u scripts/experiment_script.py
