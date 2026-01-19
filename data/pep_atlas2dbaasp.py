from qmap.benchmark import DBAASPDataset
from qmap.toolkit import compute_maximum_identity
from qmap.toolkit.utils import read_fasta
import json
import matplotlib.pyplot as plt
import numpy as np

with open("build/dbaasp.json") as f:
    dbaasp = DBAASPDataset(json.load(f))

pep_atlas = list(read_fasta(".cache/peptide_atlas.fasta").values())
print(len(pep_atlas))

np.random.seed(42)
np.random.shuffle(pep_atlas)


max_identity = compute_maximum_identity(dbaasp.sequences, pep_atlas[:500_000])

quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
quantiles_values = np.quantile(max_identity, quantiles)

for quantile, val in zip(quantiles, quantiles_values):
    print(f"Quantile {int(quantile*100)}%: {val:.3f}")

plt.hist(max_identity, bins=100, density=True, range=(0, 1))
plt.xlabel("Maximum identity to DBAASP")
plt.ylabel("Density")
plt.title("Distribution of maximum identity between Peptide Atlas and DBAASP")
plt.savefig("pep_atlas_dbaasp_identity.png")
plt.show()