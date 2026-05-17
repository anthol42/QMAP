import numpy as np
from generate_esm_embeddings import generate_esm2_embeddings
import matplotlib.pyplot as plt
from qmap.benchmark import QMAPBenchmark, DBAASPDataset
from qmap.toolkit import compute_maximum_identity
import os
from pyutils import Colors
from sklearn.linear_model import LinearRegression
import pandas as pd
import argparse
from matplotlib.collections import PolyCollection
from scipy import stats
import seaborn as sns

if __name__ == "__main__":
    colors = ['#1C7ED6', '#77DD77', '#FFB347', '#FF6961', '#17BECF', '#F4D35E']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    plt.rcParams.update({'font.size': 14})
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_he', action='store_true', help='Train on high efficiency only (MIC < 10 uM)')
    parser.add_argument('--train_me', action='store_true', help='Train on middle efficiency only (10 <= MIC <= 100 uM)')
    parser.add_argument('--train_le', action='store_true', help='Train on low efficiency only (MIC > 100 uM)')
    parser.add_argument('--rnd', action='store_true', help='Random train/test split instead of QMAP benchmark')
    args = parser.parse_args()
    dataset = (DBAASPDataset()
               .with_bacterial_targets(["Escherichia coli"])
               .with_canonical_only()
               .with_l_aa_only()
               .with_terminal_modification(False, False)
               )
    tabular = dataset.tabular(["sequence", "Escherichia coli"])
    sequences = tabular["sequence"].tolist()
    targets = np.log10(tabular["Escherichia coli"].values)

    if os.path.exists(".cache/embeddings.npy"):
        embeddings = np.load(".cache/embeddings.npy")
    else:
        embeddings = generate_esm2_embeddings(sequences)
        if not os.path.exists(".cache"):
            os.makedirs(".cache")
        np.save(".cache/embeddings.npy", embeddings)

    if args.rnd:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(sequences))
        split_point = int(0.8 * len(sequences))
        train_idx, test_idx = perm[:split_point], perm[split_point:]

        model = LinearRegression()
        model.fit(embeddings[train_idx], targets[train_idx])

        preds = model.predict(embeddings[test_idx])
        preds = [{'Escherichia coli': val.item()} for val in preds]
        eval_benchmark = dataset[test_idx]
        results = eval_benchmark.compute_metrics(preds)["Escherichia coli"]
        print(Colors.green, results, Colors.reset)

        if not os.path.exists('results'):
            os.makedirs('results')
        pd.DataFrame([results.dict()]).to_csv('results/rnd_split.csv')
        print(results.md_col, end="")
        print(results.md_row, end="")

        # Identity scatter: max train-test identity vs absolute error
        train_sequences = [sequences[i] for i in train_idx]
        test_sequences_rnd = [sequences[i] for i in test_idx]
        max_identity = compute_maximum_identity(train_sequences, test_sequences_rnd)

        test_targets = targets[test_idx]
        pred_values = np.array([p['Escherichia coli'] for p in preds])
        abs_error = np.abs(pred_values - test_targets)

        pearson_r, pearson_p = stats.pearsonr(max_identity, abs_error)
        print("Pearson r:", pearson_r)
        spearman_r, spearman_p = stats.spearmanr(max_identity, abs_error)

        slope, intercept, *_ = stats.linregress(max_identity, abs_error)
        x_line = np.linspace(max_identity.min(), max_identity.max(), 200)
        y_line = slope * x_line + intercept

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(max_identity, abs_error, alpha=0.4, s=15, rasterized=True)
        ax.plot(x_line, y_line, color=colors[3], linewidth=1.5, label='Linear fit')
        ax.set_xlabel('Max identity to nearest train sample')
        ax.set_ylabel('Absolute error (log₁₀ MIC)')
        ax.set_ylim(0, None)
        # ax.text(0.03, 0.97,
        #         f"Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\nSpearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})",
        #         transform=ax.transAxes, va='top', fontsize=9,
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.legend()
        plt.tight_layout()
        plt.savefig('../figs/rnd_identity_vs_error_linear.pdf', dpi=300)
        plt.savefig('../figs/rnd_identity_vs_error_linear.svg', dpi=300)
        plt.show()
    else:
        all_results = []
        all_results_high_eff = []
        all_results_me = []
        all_results_le = []
        for split in range(5):
            benchmark = (QMAPBenchmark(split)
                         .with_bacterial_targets(["Escherichia coli"])
                         .with_canonical_only()
                         .with_l_aa_only()
                         .with_terminal_modification(False, False)
                         .with_length_range(None, 100)
                         )
            mask = benchmark.get_train_mask(sequences)
            X_train_all = embeddings[mask]
            y_train_all = targets[mask]  # log10(MIC_uM)

            n_he = int((y_train_all < np.log10(10)).sum())
            if args.train_he:
                he = y_train_all < np.log10(10)
                X_train, y_train = X_train_all[he], y_train_all[he]
            elif args.train_me:
                me = (y_train_all >= np.log10(10)) & (y_train_all <= np.log10(100))
                X_train, y_train = X_train_all[me], y_train_all[me]
                if len(y_train) > n_he:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(y_train), n_he, replace=False)
                    X_train, y_train = X_train[idx], y_train[idx]
            elif args.train_le:
                le = y_train_all > np.log10(100)
                X_train, y_train = X_train_all[le], y_train_all[le]
                if len(y_train) > n_he:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(y_train), n_he, replace=False)
                    X_train, y_train = X_train[idx], y_train[idx]
            else:
                X_train, y_train = X_train_all, y_train_all

            if args.train_he:
                eval_benchmark = benchmark.with_efficiency_below(10.)
            elif args.train_me:
                eval_benchmark = benchmark.filter(lambda s: any(10 <= t.consensus <= 100 for t in s.targets.values()))
            elif args.train_le:
                eval_benchmark = benchmark.filter(lambda s: any(t.consensus > 100 for t in s.targets.values()))
            else:
                eval_benchmark = benchmark

            model = LinearRegression()
            model.fit(X_train, y_train)

            X_test = generate_esm2_embeddings(eval_benchmark.tabular(["sequence"])["sequence"].tolist())

            preds = model.predict(X_test)
            preds = [{'Escherichia coli': val.item()} for val in preds]
            results = eval_benchmark.compute_metrics(preds)["Escherichia coli"]
            print(Colors.green, results, Colors.reset)
            all_results.append(results)

            if not (args.train_he or args.train_me or args.train_le):
                high_eff_benchmark = benchmark.with_efficiency_below(10.)
                preds = model.predict(generate_esm2_embeddings(high_eff_benchmark.tabular(["sequence"])["sequence"].tolist()))
                preds = [{'Escherichia coli': val.item()} for val in preds]
                all_results_high_eff.append(high_eff_benchmark.compute_metrics(preds)["Escherichia coli"])

                me_benchmark = benchmark.filter(lambda s: any(10 <= t.consensus <= 100 for t in s.targets.values()))
                preds = model.predict(generate_esm2_embeddings(me_benchmark.tabular(["sequence"])["sequence"].tolist()))
                preds = [{'Escherichia coli': val.item()} for val in preds]
                all_results_me.append(me_benchmark.compute_metrics(preds)["Escherichia coli"])

                le_benchmark = benchmark.filter(lambda s: any(t.consensus > 100 for t in s.targets.values()))
                preds = model.predict(generate_esm2_embeddings(le_benchmark.tabular(["sequence"])["sequence"].tolist()))
                preds = [{'Escherichia coli': val.item()} for val in preds]
                all_results_le.append(le_benchmark.compute_metrics(preds)["Escherichia coli"])

        all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])

        # Export to pandas
        if not os.path.exists('results'):
            os.makedirs('results')
        if args.train_he:
            all_result_table.to_csv('results/train_he.csv')
        elif args.train_me:
            all_result_table.to_csv('results/train_me.csv')
        elif args.train_le:
            all_result_table.to_csv('results/train_le.csv')
        else:
            high_efficiency = pd.DataFrame([result.dict() for result in all_results_high_eff])
            me_efficiency = pd.DataFrame([result.dict() for result in all_results_me])
            le_efficiency = pd.DataFrame([result.dict() for result in all_results_le])
            all_result_table.to_csv('results/full.csv')
            high_efficiency.to_csv('results/high_efficiency.csv')
            me_efficiency.to_csv('results/me_efficiency.csv')
            le_efficiency.to_csv('results/le_efficiency.csv')

        print(all_results[0].md_col, end="")
        for results in all_results:
            print(results.md_row, end="")