#!/bin/bash
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --time=2:45:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Define parameter values (Linbranch is set to true in config)
PARAMS=(
  "--config.model.head_depth=1 --config.model.prenorm=False --config.model.head_residual=False"
  "--config.model.head_depth=2 --config.model.prenorm=False --config.model.head_residual=False"
  "--config.model.head_depth=3 --config.model.prenorm=False --config.model.head_residual=False"
  "--config.model.head_depth=1 --config.model.prenorm=True --config.model.head_residual=False"
  "--config.model.head_depth=2 --config.model.prenorm=True --config.model.head_residual=False"
  "--config.model.head_depth=3 --config.model.prenorm=True --config.model.head_residual=False"
  "--config.model.head_depth=1 --config.model.prenorm=True --config.model.head_residual=True"
  "--config.model.head_depth=2 --config.model.prenorm=True --config.model.head_residual=True"
  "--config.model.head_depth=3 --config.model.prenorm=True --config.model.head_residual=True"
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
