INPUT="../.cache/cdhit/dataset.fasta"
CLUSTERS="../.cache/cdhit/clusters"

cd-hit -i "$INPUT" -o "$CLUSTERS" -c 0.5 -n 2 -d 0 -M 30000 -T 10
