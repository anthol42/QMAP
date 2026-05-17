from EC.bert_finetune.train_test import train, _get_test_data_loader
import pandas as pd
import argparse
from qmap.benchmark import QMAPBenchmark, DBAASPDataset
import numpy as np
import torch
from pyutils import Colors
import os

parser = argparse.ArgumentParser()
parser.add_argument('--option', type=str, default='qmap')
parser.add_argument('--earlystoping', action="store_true")
parser.add_argument('--train_he', action='store_true', help='Train on high efficiency only (MIC < 10 uM)')
parser.add_argument('--train_me', action='store_true', help='Train on middle efficiency only (10 <= MIC <= 100 uM)')
parser.add_argument('--train_le', action='store_true', help='Train on low efficiency only (MIC > 100 uM)')
parser.add_argument('--rnd', action='store_true', help='Random train/val/test split instead of QMAP benchmark')

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
    elif args.rnd:
        full_data = pd.read_csv(full_path)
        rnd_test = full_data.sample(frac=0.15, random_state=42)
        remaining = full_data.drop(rnd_test.index)
        rnd_val = remaining.sample(frac=0.15 / 0.85, random_state=42)
        rnd_train = remaining.drop(rnd_val.index)

        rnd_test_benchmark = DBAASPDataset([
            {
                'id': int(row['ID']),
                'sequence': row['SEQUENCE'],
                'smiles': [],
                'nterminal': None,
                'cterminal': None,
                'bonds': [],
                'targets': {'Escherichia coli': (float(row['EC_MIC']), float(row['EC_MIC']), float(row['EC_MIC']))},
                'hemolytic_hc50': None
            }
            for _, row in rnd_test.iterrows()
        ])

        _, model, _, _, _, _ = train(0, 12,
                                     1., rnd_train,
                                     rnd_val,
                                     epochs=20,
                                     frozen_layers=0,
                                     lr=1e-5,
                                     weight_decay=3e-3,
                                     early_stopping=early_stopping
                                     )
        preds = predict(model, rnd_test['SEQUENCE'].tolist())
        results = rnd_test_benchmark.compute_metrics(preds)["Escherichia coli"]
        print(Colors.green, results, Colors.reset)

        if not os.path.exists('results'):
            os.makedirs('results')
        pd.DataFrame([results.dict()]).to_csv('results/rnd_split.csv')
        print(results.md_col, end="")
        print(results.md_row, end="")
    else:
        all_results = []
        all_results_high_complexity = []
        all_results_low_complexity = []
        all_results_high_eff = []
        all_results_me = []
        all_results_le = []

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
            if args.train_he:
                eval_benchmark = benchmark.with_efficiency_below(10.)
            elif args.train_me:
                eval_benchmark = benchmark.filter(lambda s: any(10 <= t.consensus <= 100 for t in s.targets.values()))
            elif args.train_le:
                eval_benchmark = benchmark.filter(lambda s: any(t.consensus > 100 for t in s.targets.values()))
            else:
                eval_benchmark = benchmark

            test_x = eval_benchmark.tabular(["sequence"])["sequence"].tolist()
            test_y = -np.log10(eval_benchmark.tabular(["Escherichia coli"]).values.reshape(-1))

            test_data = pd.DataFrame(dict(SEQUENCE_space=[" ".join(seq) for seq in test_x], EC_pMIC=test_y))
            mask = benchmark.get_train_mask(train_data["SEQUENCE"].values)
            valid_mask = benchmark.get_train_mask(valid_data["SEQUENCE"].values)

            split_train = train_data.loc[mask]
            n_he = (split_train['EC_MIC'] < 10).sum()
            if args.train_he:
                split_train = split_train[split_train['EC_MIC'] < 10]
            elif args.train_me:
                me = split_train[(split_train['EC_MIC'] >= 10) & (split_train['EC_MIC'] <= 100)]
                if len(me) > n_he:
                    me = me.sample(n=n_he, random_state=42)
                split_train = me
            elif args.train_le:
                le = split_train[split_train['EC_MIC'] > 100]
                if len(le) > n_he:
                    le = le.sample(n=n_he, random_state=42)
                split_train = le

            split_valid = valid_data.loc[valid_mask]
            if args.train_he:
                split_valid = split_valid[split_valid['EC_MIC'] < 10]
            elif args.train_me:
                split_valid = split_valid[(split_valid['EC_MIC'] >= 10) & (split_valid['EC_MIC'] <= 100)]
            elif args.train_le:
                split_valid = split_valid[split_valid['EC_MIC'] > 100]

            test_predict_list, model, _, _, _, _ = train(0, 12,
                                                          1., split_train,
                                                          split_valid,
                                                          epochs=5,
                                                          frozen_layers=0,
                                                          lr=1e-5,
                                                          weight_decay=3e-3,
                                                          early_stopping=early_stopping
                                                          )
            preds = predict(model, test_x)
            results = eval_benchmark.compute_metrics(preds)["Escherichia coli"]
            print(Colors.green, results, Colors.reset)
            all_results.append(results)

            if not (args.train_he or args.train_me or args.train_le):
                high_eff_benchmark = benchmark.with_efficiency_below(10.)
                preds = predict(model, high_eff_benchmark.tabular(["sequence"])["sequence"].tolist())
                all_results_high_eff.append(high_eff_benchmark.compute_metrics(preds)["Escherichia coli"])

                me_benchmark = benchmark.filter(lambda s: any(10 <= t.consensus <= 100 for t in s.targets.values()))
                preds = predict(model, me_benchmark.tabular(["sequence"])["sequence"].tolist())
                all_results_me.append(me_benchmark.compute_metrics(preds)["Escherichia coli"])

                le_benchmark = benchmark.filter(lambda s: any(t.consensus > 100 for t in s.targets.values()))
                preds = predict(model, le_benchmark.tabular(["sequence"])["sequence"].tolist())
                all_results_le.append(le_benchmark.compute_metrics(preds)["Escherichia coli"])

        all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])

        # Export to pandas
        if not os.path.exists('results'):
            os.makedirs('results')

        es_suffix = '_es' if early_stopping else ''
        if args.train_he:
            all_result_table.to_csv(f'results/train_he{es_suffix}.csv')
        elif args.train_me:
            all_result_table.to_csv(f'results/train_me{es_suffix}.csv')
        elif args.train_le:
            all_result_table.to_csv(f'results/train_le{es_suffix}.csv')
        else:
            high_efficiency = pd.DataFrame([result.dict() for result in all_results_high_eff])
            me_efficiency = pd.DataFrame([result.dict() for result in all_results_me])
            le_efficiency = pd.DataFrame([result.dict() for result in all_results_le])
            if early_stopping:
                all_result_table.to_csv('results/full_es.csv')
                high_efficiency.to_csv('results/high_efficiency_es.csv')
            else:
                all_result_table.to_csv('results/full.csv')
                high_efficiency.to_csv('results/high_efficiency.csv')
                me_efficiency.to_csv('results/me_efficiency.csv')
                le_efficiency.to_csv('results/le_efficiency.csv')
        print(all_results[0].md_col, end="")
        for results in all_results:
            print(results.md_row, end="")