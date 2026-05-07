#!/bin/bash
#SBATCH --job-name=setting_d_prime
#SBATCH --output=logs/setting_d_prime_%j.out
#SBATCH --error=logs/setting_d_prime_%j.err
#SBATCH --time=08:00:00
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
echo "Task:     Setting D prime control"
echo "======================================"

python code/setting_d_prime.py \
    --n_trials 20 \
    --T 400 \
    --n_gibbs 15 \
    --out_path results/setting_d_prime.csv

echo "Finished: $(date)"
