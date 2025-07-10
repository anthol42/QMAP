#!/bin/bash
#SBATCH --job-name=test_cluster_sampling
#SBATCH --array=0-29
#SBATCH --output=logs/test_cluster_sampling_%A_%a.out
#SBATCH --error=logs/test_cluster_sampling_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

cd ..

pwd

# Create logs directory if it doesn't exist
mkdir -p logs

# Compute output index
OUTPUT_INDEX=$((SLURM_ARRAY_TASK_ID + 30))

# Run the command
uv run make_alignments.py \
    --input=build/test_synt.fasta \
    --clusters=.cache/test_clusters_synt.clstr \
    --output=.cache/test_parts_synt/${OUTPUT_INDEX}.npy \
    --type=cluster
