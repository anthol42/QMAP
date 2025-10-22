INPUT=".cache/peptide_atlas_synt.fasta"
CLUSTERS=".cache/clusters"

cd-hit -i "$INPUT" -o "$CLUSTERS" -c 0.5 -n 2 -d 0 -M 30000 -T 10 -l 5 # There are no peptides shorter than 5 residues in the peptide atlas
