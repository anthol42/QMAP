import argparse
import os
import numpy as np
from glob import glob
from pathlib import PurePath
from tqdm import tqdm
parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str, required=True, help='Directory with alignments parts. (Numpy matrices .npy')
parser.add_argument('--output', type=str, required=True, help='Output file (.npy)')
parser.add_argument("--min_samples", type=int, default=500_000, help="Minimum number of samples to subsample from each bin. Default: 500000")
parser.add_argument("--max_idx", type=int, default=None, help="Max index to merge. Default: None")

def get_idx(file_path: str) -> int:
    name = PurePath(file_path).name
    return int(name.split('.')[0])

def subsample(labels, min_samples: int = 500_000) -> np.ndarray:
    """
    Convert a gaussian distribution of labels to a uniform distribution by invert sampling.
    :param labels: The labels to sample from (between 0 and 1)
    :param min_samples: The minimum number of samples. Under this, no sampling is done
    :return: A mask of the labels that are sampled uniformly
    """
    mask = np.zeros_like(labels, dtype=bool)
    for i in tqdm(range(100), desc="Subsambling"):
        low, high = i / 100, (i + 1) / 100
        bin_mask = np.logical_and(labels >= low, labels <= high)
        if bin_mask.sum() < min_samples:
            mask = np.logical_or(mask, bin_mask)
        else:
            sampled = np.random.choice(np.where(bin_mask)[0], min_samples, replace=False)
            mask[sampled] = True

    return mask

if __name__ == '__main__':
    args = parser.parse_args()


    files = glob(os.path.join(args.dir, '*.npy'))
    if not files:
        raise ValueError(f"No .npy files found in directory {args.dir}")

    # Sort files by index extracted from the filename
    files.sort(key=get_idx)

    # Filter files if max_idx is set
    if args.max_idx:
        files = [path for path in files if get_idx(path) <= args.max_idx]

    # Load and concatenate all alignments
    alignments = []
    for file_path in files:
        data = np.load(file_path)
        alignments.append(data)

    alignments = np.concatenate(alignments, axis=0)

    # Now, drop duplicates based on the first two columns (src, dst)
    # This can happen when there are collisions in the random sampling
    alignments = np.unique(alignments, axis=0)

    # Subsamble to get something like a uniform distribution
    mask = subsample(alignments[:, 2], min_samples=args.min_samples)
    alignments = alignments[mask]

    # Save the merged alignments
    np.save(args.output, alignments)
    print(f"Merged alignments saved to {args.output}")