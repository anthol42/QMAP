#!/bin/bash
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --time=2:45:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

PARAMS=(
  "--config.training.orthogonality=0.05 --config.training.diversity=0.001"

  "--config.training.orthogonality=0.05 --config.training.diversity=0.001 --config.training.var=0.1"

  "--config.training.orthogonality=0.05 --config.training.diversity=0.001 --config.training.smoothness=0.01"

  "--config.training.orthogonality=0.05 --config.training.diversity=0.001 --config.training.smoothness=0.01 --config.training.var=0.1"
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
