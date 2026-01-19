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
               .with_hc50()
               .with_canonical_only()
               .with_l_aa_only()
               .with_terminal_modification(False, False)
               )
    tabular = dataset.tabular(["sequence", "hc50"])
    sequences = tabular["sequence"].tolist()
    targets = np.log10(tabular["hc50"].values)

    if os.path.exists(".cache/embeddings.npy"):
        embeddings = np.load(".cache/embeddings.npy")
    else:
        embeddings = generate_esm2_embeddings(sequences)
        if not os.path.exists(".cache"):
            os.makedirs(".cache")
        np.save(".cache/embeddings.npy", embeddings)

    all_results = []
    for split in range(5):
        benchmark = (QMAPBenchmark(split)
                     .with_hc50()
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
        preds = [{'hc50': val.item()} for val in preds]
        results = benchmark.compute_metrics(preds)["hc50"]
        print(Colors.green, results, Colors.reset)
        all_results.append(results)

    all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])

    # Export to pandas
    if not os.path.exists('results'):
        os.makedirs('results')
    all_result_table.to_csv('results/full.csv')

    print(all_results[0].md_col, end="")
    for results in all_results:
        print(results.md_row, end="")