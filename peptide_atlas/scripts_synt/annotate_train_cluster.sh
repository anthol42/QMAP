#!/bin/bash
#SBATCH --job-name=train_cluster_sampling
#SBATCH --array=0-99
#SBATCH --output=logs/train_cluster_sampling_%A_%a.out
#SBATCH --error=logs/train_cluster_sampling_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

cd ..

pwd

# Create logs directory if it doesn't exist
mkdir -p logs

# Compute output index
OUTPUT_INDEX=$((SLURM_ARRAY_TASK_ID + 100))

# Run the command
uv run make_alignments.py \
    --input=build/train_synt.fasta \
    --clusters=.cache/train_clusters_synt.clstr \
    --output=.cache/train_parts_synt/${OUTPUT_INDEX}.npy \
    --type=cluster
