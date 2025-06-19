import argparse
import os.path
from tqdm import tqdm
import numpy as np
from utils import compute_identity

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, required=True, help='fasta file')
parser.add_argument('--id_range', type=str, required=True, help='id range')

if __name__ == '__main__':
    args = parser.parse_args()
    start_idx, end_idx = map(int, args.id_range.split('-'))

    with open(args.input, 'r') as f:
        lines = f.readlines()
    # Load all sequences in a list[(id, sequence)]
    sequences = []
    for line in lines:
        if line.startswith('>'):
            id_ = line[1:].strip()
        else:
            sequence = line.strip()
            sequences.append((id_, sequence))

    identities = np.full((end_idx - start_idx, len(sequences)), np.nan)
    row_ids = []
    for i, sequence in tqdm(enumerate(sequences[start_idx:end_idx]), total=end_idx - start_idx):
        id_, seq = sequence
        for j, target_sequence in enumerate(sequences):
            target_id, target_seq = target_sequence
            if id_ == target_id:
                identity = 1.
            else:
                identity = compute_identity(seq, target_seq)
            identities[i, j] = identity
        row_ids.append(id_)

    # Save the identities matrix to a file
    if not os.path.exists("matrix_parts"):
        os.mkdir("matrix_parts")
    output_file = f'matrix_parts/{start_idx}-{end_idx}.npy'
    np.save(output_file, identities)
    # Save the row IDs to a file
    with open(f'matrix_parts/{start_idx}-{end_idx}.txt', 'w') as f:
        for row_id in row_ids:
            f.write(f"{row_id}\n")

# Total of 19038 sequences in the dataset