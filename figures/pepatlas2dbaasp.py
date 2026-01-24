from qmap.benchmark import DBAASPDataset
from qmap.toolkit import compute_maximum_identity
from qmap.toolkit.utils import read_fasta
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#1C7ED6', '#77DD77', '#FFB347', '#FF6961', '#17BECF', '#F4D35E', "#ADFF2F", "#FF8200"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)
plt.rcParams.update({'font.size': 14})


dbaasp = DBAASPDataset()

pep_atlas = list(read_fasta("../data/.cache/peptide_atlas.fasta").values())
print(len(pep_atlas))

np.random.seed(42)
np.random.shuffle(pep_atlas)


max_identity = compute_maximum_identity(dbaasp.sequences, pep_atlas)

quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
quantiles_values = np.quantile(max_identity, quantiles)

with open('figs/pep_atlas_dbaasp_identity.txt', 'w') as f:
    for quantile, val in zip(quantiles, quantiles_values):
        print(f"Quantile {int(quantile*100)}%: {val:.3f}")
        f.write(f'{int(quantile*100)};{val:.3f}\n')

plt.hist(max_identity, bins=100, density=True, range=(0, 1))
plt.xlabel("Maximum identity to DBAASP")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("figs/pep_atlas_dbaasp_identity.svg")
plt.savefig("figs/pep_atlas_dbaasp_identity.pdf")