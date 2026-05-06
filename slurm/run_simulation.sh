#!/bin/bash
#SBATCH --job-name=hier_bandit_sim
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --array=0-3                   # 4 settings: A B C D
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Note: the simulation is CPU-only (NumPy). The "gpu" partition is
# requested only because Oscar's batch nodes have less memory; switch
# to "batch" or "compute" if memory is not an issue.

module purge
module load python/3.11

source /oscar/home/hzhan382/bayesiantest/bandit_env/bin/activate

# Map array index to setting
SETTINGS=(A B C D)
SETTING=${SETTINGS[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Task:     $SLURM_ARRAY_TASK_ID -> Setting $SETTING"
echo "Node:     $HOSTNAME"
echo "Start:    $(date)"
echo "======================================"

cd /oscar/home/hzhan382/bayesiantest
mkdir -p logs results/figures results/checkpoints results/tables

python code/simulation.py \
    --setting   $SETTING \
    --n_trials  20 \
    --T         400 \
    --n_gibbs   15 \
    --seed      42 \
    --outdir    results

echo "Finished: $(date)"
