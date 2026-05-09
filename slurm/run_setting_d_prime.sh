#!/bin/bash
#SBATCH --job-name=setting_d_prime
#SBATCH --output=logs/setting_d_prime_%j.out
#SBATCH --error=logs/setting_d_prime_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

# CPU job. Submit from the repository root after creating bandit_env.
module purge
module load python/3.11

cd "$SLURM_SUBMIT_DIR"
source bandit_env/bin/activate
mkdir -p logs results/figures results/checkpoints results/tables

python code/setting_d_prime.py \
    --n_trials 20 \
    --T 400 \
    --n_gibbs 15 \
    --out_path results/setting_d_prime.csv
