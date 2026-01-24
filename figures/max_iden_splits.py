import os
import numpy as np
import matplotlib.pyplot as plt
from qmap.toolkit import compute_maximum_identity
from qmap import train_test_split
from qmap import DBAASPDataset

if not os.path.exists("figs"):
    os.makedirs("figs")

COLORS = ['#1C7ED6', '#77DD77', '#FFB347', '#FF6961', '#17BECF', '#F4D35E', "#ADFF2F", "#FF8200"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)

def plot_max_iden_dist(train_seq, test_seq, name):
    max_iden = compute_maximum_identity(train_seq, test_seq)

    # Plot
    print(f"Max identity: {np.max(max_iden)}")
    print(f"Mean identity: {np.mean(max_iden)}")
    print(f"Median identity: {np.median(max_iden)}")
    print("Quantiles:")
    with open(f'figs/{name}.txt', 'w') as f:
        for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            print(f"- {q:.2f} quantile: {np.quantile(max_iden, q)}")
            f.write(f"{q:.2f} quantile: {np.quantile(max_iden, q):.3f}\n")

    plt.hist(max_iden, bins=50, range=(0, 1), density=True)
    plt.xlabel("Identity")
    plt.ylabel("Density")
    # plt.title("Highest identity between each test sequence and all train sequences")
    plt.tight_layout()
    plt.savefig(f"figs/{name}.pdf")
    plt.savefig(f"figs/{name}.svg")
    plt.close()

sequences = DBAASPDataset().sequences

# Homology based split:
train_seq, test_seq = train_test_split(sequences, test_size=0.2, random_state=404)
plot_max_iden_dist(train_seq, test_seq, "homology_split")

# Random split:
np.random.seed(404)
np.random.shuffle(sequences)
n_test = int(0.2 * len(sequences))
test_seq = sequences[:n_test]
train_seq = sequences[n_test:]
plot_max_iden_dist(train_seq, test_seq, "random_split")