INPUT="../.cache/mmseqs/dataset.fasta"
DB="../.cache/mmseqs/peptidesDB"
CLUSTERS="../.cache/mmseqs/clusters"
TMP="../.cache/mmseqs/tmp"

# Create database
mmseqs createdb "$INPUT" "$DB"

# Cluster with short-peptide optimized settings
mmseqs linclust "$DB" "$CLUSTERS" "$TMP" \
    --min-seq-id 0.5 \
    --cov-mode 1 -c 0.8 \
    --cluster-mode 2 \
    --kmer-per-seq 80 \
    --comp-bias-corr 0

# Export representative sequences of clusters
#mmseqs createtsv "$DB" "$DB" "$RESULT" clusters.tsv
#mmseqs createseqfiledb "$DB" "$RESULT" "$RESULT"_seq
#mmseqs result2flat "$DB" "$DB" "$RESULT" "$RESULT".fasta
#
#echo "Clustering done. Output:"
#echo "- Cluster map: clusters.tsv"
#echo "- Clustered sequences: ${RESULT}.fasta"