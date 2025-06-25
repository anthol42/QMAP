"""
To keep the same settings as in the other experiments, we will subsample the dataset to keep only
the sequences that are between 5 and 50 amino acids long.
"""
from pathlib import PurePath
import os
import sys
os.chdir(PurePath(__file__).parent.parent)
sys.path.append(os.getcwd())
import numpy as np
from pyutils import progress

from split_utils import read_fasta, Identity, compute_identity_stats
import matplotlib.pyplot as plt

fasta = read_fasta("../data/build/dataset.fasta")
dataset = [int(id_) for id_, seq in fasta if 5 <= len(seq) <= 50]

# Random split
indices = np.arange(len(dataset))
np.random.shuffle(indices)
test_indices = indices[:int(len(dataset) * 0.2)]
train_indices = indices[int(len(dataset) * 0.2):]
train_ids = [dataset[i] for i in train_indices]
test_ids = [dataset[i] for i in test_indices]

# Compute the statistics on the identity between the two splits
identity_calculator = Identity()

identities, true_train_set = compute_identity_stats(train_ids, test_ids, identity_calculator=identity_calculator)
if np.isnan(identities).any():
    print("Warning: There are NaN values in the identities array. This may indicate missing data for some sequences.")

# Print statistics about the identities (Max identity, mean, median, quantiles)
print(f"Max identity: {np.max(identities)}")
print(f"Mean identity: {np.mean(identities)}")
print(f"Median identity: {np.median(identities)}")
print("Quantiles:")
for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"- {q:.2f} quantile: {np.quantile(identities, q)}")

print("Number of sequences in the training set that are not similar to any test sequence (identity > 0.5):")
print(np.sum(true_train_set))
plt.hist(identities, bins=50)
plt.xlabel("Identity")
plt.ylabel("Frequency")
plt.show()

