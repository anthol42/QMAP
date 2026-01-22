from EC.bert_finetune.train_test import train, _get_test_data_loader
import pandas as pd
import argparse
from qmap.benchmark import QMAPBenchmark
import numpy as np
import torch
from pyutils import Colors
import os

parser = argparse.ArgumentParser()
parser.add_argument('--option', type=str, default='qmap')
parser.add_argument('--earlystoping', action="store_true")

def predict(model, X):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = []
    model.eval()
    test_data = pd.DataFrame(dict(
        pd.DataFrame(dict(SEQUENCE_space=[" ".join(seq) for seq in X], EC_pMIC=np.zeros(len(X))))
    ))
    test_loader = _get_test_data_loader(500, test_data)
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            predict_MIC, _ = model(b_input_ids, attention_mask=b_input_mask)

            preds.append(predict_MIC.cpu().numpy())

    return [{"Escherichia coli": -pred.item()} for pred in np.concatenate(preds)]

if __name__ == '__main__':
    args = parser.parse_args()
    early_stopping = args.earlystoping
    full_path = f'data/EC.csv'
    train_path = f"data/train-EC.csv"
    test_path = f"data/test-EC.csv"
    if args.option == 'original':
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        test_predict_list, model, _, _, _, _ = train(0, 12,
                                                     1., train_data,
                                                     test_data,
                                                     epochs=20,
                                                     frozen_layers=0,
                                                     lr=1e-5,
                                                     weight_decay=3e-3
                                                     )
    else:
        all_results = []
        all_results_high_complexity = []
        all_results_low_complexity = []
        all_results_high_eff = []

        train_data = pd.read_csv(full_path)
        valid_data = pd.read_csv(test_path)
        for split in range(5):
            benchmark = (QMAPBenchmark(split)
                         .with_bacterial_targets(["Escherichia coli"])
                         .with_canonical_only()
                         .with_l_aa_only()
                         .with_terminal_modification(False, False)
                         .with_length_range(None, 100)
                         )
            test_x = benchmark.tabular(["sequence"])["sequence"].tolist()
            test_y = -np.log10(benchmark.tabular(["Escherichia coli"]).values.reshape(-1))

            test_data = pd.DataFrame(dict(SEQUENCE_space=[" ".join(seq) for seq in test_x], EC_pMIC=test_y))
            mask = benchmark.get_train_mask(train_data["SEQUENCE"].values)
            valid_mask = benchmark.get_train_mask(valid_data["SEQUENCE"].values)
            test_predict_list, model, _, _, _, _ = train(0, 12,
                                                          1., train_data.loc[mask],
                                                          valid_data.loc[valid_mask],
                                                          epochs=5,
                                                          frozen_layers=0,
                                                          lr=1e-5,
                                                          weight_decay=3e-3,
                                                          early_stopping=early_stopping
                                                          )
            preds = predict(model, test_x)
            results = benchmark.compute_metrics(preds)["Escherichia coli"]
            print(Colors.green, results, Colors.reset)
            all_results.append(results)


            high_eff_benchmark = benchmark.with_efficiency_below(10.)
            preds = predict(model, high_eff_benchmark.tabular(["sequence"])["sequence"].tolist())
            all_results_high_eff.append(high_eff_benchmark.compute_metrics(preds)["Escherichia coli"])

        all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])
        high_efficiency = pd.DataFrame([result.dict() for result in all_results_high_eff])

        # Export to pandas
        if not os.path.exists('results'):
            os.makedirs('results')

        if early_stopping:
            all_result_table.to_csv(f'results/full_es.csv')
            high_efficiency.to_csv('results/high_efficiency_es.csv')
        else:
            all_result_table.to_csv(f'results/full.csv')
            high_efficiency.to_csv('results/high_efficiency.csv')
        print(all_results[0].md_col, end="")
        for results in all_results:
            print(results.md_row, end="")