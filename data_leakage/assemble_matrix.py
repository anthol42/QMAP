import os.path
import numpy as np
from glob import glob
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, required=True, help='The directory containing the parts of the global matrix')
PARTS_PATH = parser.parse_args().input

if __name__ == "__main__":
    # Get all parts
    parts = glob(f"{PARTS_PATH}/*.npy")
    parts.sort(key=lambda x: int(Path(x).name.split("-")[0]))

    # Load all parts and concatenate them
    matrices = []
    row_ids = []
    for part in parts:
        matrix = np.load(part)
        matrices.append(matrix)
        with open(f"{part.split('.')[0]}.txt", 'r') as f:
            ids = f.readlines()
            ids = [id_.strip() for id_ in ids]
            row_ids.extend(ids)
    assert len(set(row_ids)) == len(row_ids), "There are duplicate row IDs!"
    full_matrix = np.concatenate(matrices, axis=0)
    full_matrix = full_matrix[:full_matrix.shape[1]]

    assert len(full_matrix) == len(row_ids), "There is not the same number of rows in the matrix and row IDs!"

    # Save the full matrix and row IDs
    if not os.path.exists(".cache"):
        os.makedirs(".cache")

    np.save(".cache/identity_matrix.npy", full_matrix)
    with open(".cache/row_ids.txt", 'w') as f:
        for row_id in row_ids:
            f.write(f"{row_id}\n")