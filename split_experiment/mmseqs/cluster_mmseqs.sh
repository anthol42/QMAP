INPUT=".cache/mmseqs/dataset.fasta"
DB=".cache/mmseqs/peptidesDB"
CLUSTERS=".cache/mmseqs/clusters"
TMP=".cache/mmseqs/tmp"

# Create database
mmseqs createdb "$INPUT" "$DB"

# Cluster with short-peptide optimized settings
mmseqs  "$DB" "$CLUSTERS" "$TMP" \
    --min-seq-id 0.55 \
    --cov-mode 0 -c 0. \
    --cluster-mode 1 \
    --kmer-per-seq 100000 \
    --comp-bias-corr 0

# Export representative sequences of clusters
mmseqs createtsv "$DB" "$DB" "$CLUSTERS" ".cache/mmseqs/clusters.tsv"
