#!/bin/bash
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --time=70:00:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs


echo "Starting job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID"
echo "Running on node: $HOSTNAME"
echo "Start time: $(date)"

# Run the command
uv run main.py --experiment=experiment1_hsearch --config=configs/ESM_35M.yml --fract=0.1 --verbose=1 --n_trials=18

echo "End time: $(date)"
echo "Job completed"
