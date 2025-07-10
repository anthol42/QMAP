#!/bin/bash
#SBATCH --job-name=compute_identity
#SBATCH --array=0-29
#SBATCH --output=logs/compute_identity_%A_%a.out
#SBATCH --error=logs/compute_identity_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

cd ..

pwd

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the command
uv run make_alignments.py --input=build/test_synt.fasta --output=.cache/test_parts_synt/${SLURM_ARRAY_TASK_ID}.npy