"""
Around 1h to run with rtx 3090 gpu and intel i7 13th gen
"""
import os
from qmap.toolkit import train_test_split, compute_maximum_identity
from qmap.benchmark import Target
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from qmap import DBAASPDataset
from scipy.stats import pearsonr

from utils import generate_esm2_embeddings


COLORS = ['#1C7ED6', '#77DD77', '#FFB347', '#FF6961', '#17BECF', '#F4D35E', "#ADFF2F", "#FF8200"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)
plt.rcParams.update({'font.size': 14})


if not os.path.exists('.cache'):
    os.makedirs('.cache')

dataset = DBAASPDataset()
sequences = dataset.sequences

if not os.path.exists(".cache/esm2_embeddings.npy"):
    embs = generate_esm2_embeddings(sequences, batch_size=128)
    np.save(".cache/esm2_embeddings.npy", embs)
else:
    embs = np.load(".cache/esm2_embeddings.npy")


thresholds = [0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.9, 0.99]
if os.path.exists(".cache/pcc_trains.npy") and os.path.exists(".cache/pcc_tests.npy") and os.path.exists(".cache/max_idens_median.npy") and os.path.exists(".cache/max_idens_99th.npy"):
    pcc_trains = np.load(".cache/pcc_trains.npy")
    pcc_tests = np.load(".cache/pcc_tests.npy")
    max_idens_median = np.load(".cache/max_idens_median.npy")
    max_idens_99th = np.load(".cache/max_idens_99th.npy")
else:
    # Try different thresholds
    pcc_trains = np.empty((len(thresholds), 5))
    pcc_tests = np.empty((len(thresholds), 5))
    max_idens_median = np.empty((len(thresholds), 5))
    max_idens_99th = np.empty((len(thresholds), 5))
    for i, threshold in enumerate(thresholds):
        for j, seed in enumerate([1, 3, 7, 12, 404]):
            train_seq, test_seq, train_embs, test_embs, train_samples, test_samples = train_test_split(
                sequences,
                embs,
                list(dataset),
                threshold=threshold,
                random_state=seed,
            )
            print(f"Train size: {len(train_seq)}, Test size: {len(test_seq)}")

            train_y = np.array([sample.targets.get('Escherichia coli', Target('Escherichia coli', np.nan, np.nan, np.nan)).consensus for sample in train_samples])
            train_y[np.isinf(train_y)] = np.nan
            mask = ~np.isnan(train_y)
            train_y = np.log10(train_y[mask])

            train_embs = np.array(train_embs)[mask]

            test_y = np.array([sample.targets.get('Escherichia coli', Target('Escherichia coli', np.nan, np.nan, np.nan)).consensus for sample in test_samples])
            test_y[np.isinf(test_y)] = np.nan
            mask = ~np.isnan(test_y)
            test_y = np.log10(test_y[mask])
            test_embs = np.array(test_embs)[mask]

            # Compute the maximum identity distribution
            max_iden = compute_maximum_identity(train_seq, test_seq)
            median_iden = np.median(max_iden)
            max_idens_median[i, j] = median_iden
            max_idens_99th[i, j] = np.quantile(max_iden, 0.99)
            # Make a linear regression model
            reg = LinearRegression()
            reg.fit(train_embs, train_y)

            pcc_train = pearsonr(reg.predict(train_embs), train_y).statistic
            pcc_test = pearsonr(reg.predict(test_embs), test_y).statistic
            pcc_trains[i, j] = pcc_train
            pcc_tests[i, j] = pcc_test
            print(f'[it {j}]', threshold, pcc_train, pcc_test, median_iden)
    np.save(".cache/pcc_trains.npy", pcc_trains)
    np.save(".cache/pcc_tests.npy", pcc_tests)
    np.save(".cache/max_idens_median.npy", max_idens_median)
    np.save(".cache/max_idens_99th.npy", max_idens_99th)


plt.plot(thresholds, np.mean(pcc_tests, axis=1), label='Test PCC', marker='o')
plt.fill_between(thresholds, np.max(pcc_tests, axis=1), np.min(pcc_tests, axis=1), color=COLORS[0], alpha=0.3)

# Add a vertical line at the threshold of 0.6
plt.axvline(x=0.60, color=COLORS[1], linestyle='--', label='Threshold 0.60')
plt.xlabel('Threshold (%)')
plt.ylabel('PCC')
plt.xticks(thresholds, [f'{t:.2f}'[2:] for t in thresholds])
plt.grid()
plt.tight_layout()
plt.savefig("figs/pcc_vs_threshold.svg")
plt.savefig("figs/pcc_vs_threshold.pdf")

with open('figs/pcc_vs_threshold.txt', 'w') as f:
    for threshold, median_iden in zip(thresholds, max_idens_median):
        f.write(f'Threshold: {threshold:.2f}, Median max identity: {median_iden.mean():.2f}±{median_iden.std():.2f}\n')
