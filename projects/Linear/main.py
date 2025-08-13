import numpy as np
from generate_esm_embeddings import generate_esm2_embeddings
import json
from qmap.benchmark import QMAPBenchmark
import os
from pyutils import Colors
from sklearn.linear_model import LinearRegression
import pandas as pd

if __name__ == "__main__":
    with open('../../data/build/dataset.json', 'r') as f:
        dataset = json.load(f)
    sequences = [sample['Sequence'] for sample in dataset if len(sample['Sequence']) < 100]
    targets = [sample["Targets"].get("Escherichia coli", (None, None, None))[0] for sample in dataset if len(sample['Sequence']) < 100]
    sequences = [sequence for target, sequence in zip(targets, sequences) if target is not None]
    targets = np.log10(np.array([target for target in targets if target is not None]))
    if os.path.exists(".cache/embeddings.npy"):
        embeddings = np.load(".cache/embeddings.npy")
    else:
        embeddings = generate_esm2_embeddings(sequences)
        if not os.path.exists(".cache"):
            os.makedirs(".cache")
        np.save(".cache/embeddings.npy", embeddings)

    all_results = []
    all_results_high_complexity = []
    all_results_low_complexity = []
    all_results_high_eff = []
    for split in range(5):
        for threshold in [55, 60]:
            benchmark = QMAPBenchmark(split, threshold,
                                      species_subset=['Escherichia coli']
                                      )
            mask = benchmark.get_train_mask(sequences)
            X_train = embeddings[mask]
            y_train = targets[mask]

            model = LinearRegression()
            model.fit(X_train, y_train)

            X_test = generate_esm2_embeddings(benchmark.inputs)

            preds = model.predict(X_test)

            results = benchmark.compute_metrics(preds)
            print(Colors.green, results, Colors.reset)
            all_results.append(results)

            high_comp_benchmark = benchmark.high_complexity
            preds = model.predict(generate_esm2_embeddings(high_comp_benchmark.inputs))
            all_results_high_complexity.append(high_comp_benchmark.compute_metrics(preds))

            low_comp_benchmark = benchmark.low_complexity
            preds = model.predict(generate_esm2_embeddings(low_comp_benchmark.inputs))
            all_results_low_complexity.append(low_comp_benchmark.compute_metrics(preds))

            high_eff_benchmark = benchmark.high_efficiency
            preds = model.predict(generate_esm2_embeddings(high_eff_benchmark.inputs))
            all_results_high_eff.append(high_eff_benchmark.compute_metrics(preds))

    all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])
    high_complexity = pd.DataFrame([result.dict() for result in all_results_high_complexity])
    low_complexity = pd.DataFrame([result.dict() for result in all_results_low_complexity])
    high_efficiency = pd.DataFrame([result.dict() for result in all_results_high_eff])

    # Export to pandas
    if not os.path.exists('.cache'):
        os.makedirs('.cache')
    all_result_table.to_csv('.cache/full.csv')
    high_complexity.to_csv('.cache/high_complexity.csv')
    low_complexity.to_csv('.cache/low_complexity.csv')
    high_efficiency.to_csv('.cache/high_efficiency.csv')

    print(all_results[0].md_col, end="")
    for results in all_results:
        print(results.md_row, end="")