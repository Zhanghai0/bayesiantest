#!/bin/bash
#SBATCH --job-name=m_ablation
#SBATCH --output=logs/m_ablation_%j.out
#SBATCH --error=logs/m_ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

# CPU job. Submit from the repository root after creating bandit_env.
module purge
module load python/3.11

cd "$SLURM_SUBMIT_DIR"
source bandit_env/bin/activate
mkdir -p logs results/figures results/checkpoints results/tables

python code/m_ablation.py \
    --M_values 15 50 100 \
    --n_trials 20 \
    --T 400 \
    --out_path results/m_ablation.csv
