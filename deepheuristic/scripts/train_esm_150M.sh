#!/bin/bash
#SBATCH --job-name=%x
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Define parameter values
PARAMS=(64 128 256 512)

# Get the parameter for this array task
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID"
echo "Using proj_dim parameter: $PARAM"
echo "Running on node: $HOSTNAME"
echo "Start time: $(date)"

# Run the command
uv run main.py --experiment=experiment1 --config=configs/ESM_150M.yml --fract-0.1 --config.model.proj_dim=$PARAM

echo "End time: $(date)"
echo "Job completed"