import argparse
import os
import numpy as np
from glob import glob
from pathlib import PurePath
parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str, required=True, help='Directory with alignments parts. (Numpy matrices .npy')
parser.add_argument('--output', type=str, required=True, help='Output file (.npy)')

def get_idx(file_path: str) -> int:
    name = PurePath(file_path).name
    return int(name.split('.')[0])

if __name__ == '__main__':
    args = parser.parse_args()


    files = glob(os.path.join(args.dir, '*.npy'))
    if not files:
        raise ValueError(f"No .npy files found in directory {args.dir}")

    # Sort files by index extracted from the filename
    files.sort(key=get_idx)

    # Load and concatenate all alignments
    alignments = []
    for file_path in files:
        data = np.load(file_path)
        alignments.append(data)

    alignments = np.concatenate(alignments, axis=0)

    # Now, drop duplicates based on the first two columns (src, dst)
    # This can happen when there are collisions in the random sampling
    alignments = np.unique(alignments, axis=0)

    # Save the merged alignments
    np.save(args.output, alignments)
    print(f"Merged alignments saved to {args.output}")