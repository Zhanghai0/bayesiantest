#!/bin/bash
#SBATCH --job-name=stan_validation
#SBATCH --output=logs/stan_%j.out
#SBATCH --error=logs/stan_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

cd /oscar/home/hzhan382/bayesiantest
source bandit_env/bin/activate

mkdir -p logs results/figures results/tables

python code/stan_validation.py
