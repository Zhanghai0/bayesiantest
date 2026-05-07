#!/bin/bash
#SBATCH --job-name=m_ablation
#SBATCH --output=logs/m_ablation_%j.out
#SBATCH --error=logs/m_ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

set -euo pipefail

module purge
module load python/3.11

cd /oscar/home/hzhan382/bayesiantest
source bandit_env/bin/activate

mkdir -p logs results/figures results/checkpoints results/tables

echo "======================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $HOSTNAME"
echo "Start:    $(date)"
echo "Task:     Setting A M-ablation"
echo "======================================"

python code/m_ablation.py \
    --M_values 15 50 100 \
    --n_trials 20 \
    --T 400 \
    --out_path results/m_ablation.csv

echo "Finished: $(date)"
