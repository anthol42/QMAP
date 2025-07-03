#!/bin/bash
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --time=6:00:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Define parameter values
PARAMS=(
  "--config.training.loss=BCE --config.model.activation_agglomeration=mult"
  "--config.training.loss=BCE --config.model.activation_agglomeration=abs_diff"
  "--config.training.loss=BCE --config.model.activation_agglomeration=cat"
  "--config.training.loss=MSE --config.model.activation_agglomeration=mult"
  "--config.training.loss=MSE --config.model.activation_agglomeration=abs_diff"
  "--config.training.loss=MSE --config.model.activation_agglomeration=cat"
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
