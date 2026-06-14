#!/bin/bash
#SBATCH --job-name=orchid_downscale
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH -o /home/users/sbarnett/documents/swellbound/slurm_scripts/logs/output.%j.out # STDOUT
#SBATCH -e /home/users/sbarnett/documents/swellbound/slurm_scripts/logs/output.%j.err
#SBATCH --time=00:10:00              # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=8GB                    # Memory request
#SBATCH --cpus-per-task=4          # CPU cores per task


# 1. Set up the environment
module load jaspy/3.12/v20250704
source ~/my_venvs/driftnet_venv/bin/activate

# 2. Navigate to the directory containing your script
cd ~/documents/driftnet/

export LD_LIBRARY_PATH=/apps/jasmin/jaspy/miniforge_envs/jaspy3.12/mf3-25.3.0-3/envs/jaspy3.12-mf3-25.3.0-3-v20250704/lib:$LD_LIBRARY_PATH

python -u scripts/ml/experiment_script.py