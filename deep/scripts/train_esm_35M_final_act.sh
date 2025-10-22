#!/bin/bash
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --time=11:00:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Define parameter values (Linbranch is set to true in config)
PARAMS=(
  "--config.training.activation_type=Linear"
  "--config.training.activation_type=Identity"
)

# Get the parameter for this array task
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID"
echo "Using '$PARAM'"
echo "Running on node: $HOSTNAME"
echo "Start time: $(date)"

# Run the command
uv run main.py --experiment=experiment1 --config=configs/ESM_35M.yml --fract=0.1 --verbose=2 $PARAM

echo "End time: $(date)"
echo "Job completed"
