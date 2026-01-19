import numpy as np
from generate_esm_embeddings import generate_esm2_embeddings
import json
from qmap.benchmark import QMAPBenchmark, DBAASPDataset
import os
from pyutils import Colors
from sklearn.linear_model import LinearRegression
import pandas as pd

if __name__ == "__main__":
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

    all_results = []
    all_results_high_eff = []
    for split in range(5):
        benchmark = (QMAPBenchmark(split)
                     .with_bacterial_targets(["Escherichia coli"])
                     .with_canonical_only()
                     .with_l_aa_only()
                     .with_terminal_modification(False, False)
                     .with_length_range(None, 100)
                     )
        mask = benchmark.get_train_mask(sequences)
        X_train = embeddings[mask]
        y_train = targets[mask]

        model = LinearRegression()
        model.fit(X_train, y_train)

        X_test = generate_esm2_embeddings(benchmark.tabular(["sequence"])["sequence"].tolist())

        preds = model.predict(X_test)
        preds = [{'Escherichia coli': val.item()} for val in preds]
        results = benchmark.compute_metrics(preds)["Escherichia coli"]
        print(Colors.green, results, Colors.reset)
        all_results.append(results)

        high_eff_benchmark = benchmark.with_efficiency_below(10.)
        preds = model.predict(generate_esm2_embeddings(high_eff_benchmark.tabular(["sequence"])["sequence"].tolist()))
        preds = [{'Escherichia coli': val.item()} for val in preds]
        all_results_high_eff.append(high_eff_benchmark.compute_metrics(preds)["Escherichia coli"])

    all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])
    high_efficiency = pd.DataFrame([result.dict() for result in all_results_high_eff])

    # Export to pandas
    if not os.path.exists('results'):
        os.makedirs('results')
    all_result_table.to_csv('results/full.csv')
    high_efficiency.to_csv('results/high_efficiency.csv')

    print(all_results[0].md_col, end="")
    for results in all_results:
        print(results.md_row, end="")