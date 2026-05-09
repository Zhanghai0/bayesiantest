#!/bin/bash
#SBATCH --job-name=hier_bandit_sim
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

# CPU job. Submit from the repository root after creating bandit_env.
# If Oscar requires an explicit partition, add the appropriate CPU partition.
module purge
module load python/3.11

cd "$SLURM_SUBMIT_DIR"
source bandit_env/bin/activate
mkdir -p logs results/figures results/checkpoints results/tables

SETTINGS=(A B C D)
SETTING=${SETTINGS[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Task:     $SLURM_ARRAY_TASK_ID -> Setting $SETTING"
echo "Node:     $HOSTNAME"
echo "Start:    $(date)"
echo "======================================"

python code/simulation.py \
    --setting   "$SETTING" \
    --n_trials  20 \
    --T         400 \
    --n_gibbs   15 \
    --seed      42 \
    --outdir    results

echo "Finished: $(date)"
