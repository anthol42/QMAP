#!/bin/bash
#SBATCH --job-name=compute_identity
#SBATCH --array=0-99
#SBATCH --output=logs/compute_identity_%A_%a.out
#SBATCH --error=logs/compute_identity_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate the range values
START=$((200 * SLURM_ARRAY_TASK_ID))
if [ $SLURM_ARRAY_TASK_ID -eq 99 ]; then
    END=19038
else
    END=$((200 * SLURM_ARRAY_TASK_ID + 200))
fi

# Run the command
uv run ground_truth/compute_identity.py --input=../data/build/dataset.fasta --input=${START}-${END}