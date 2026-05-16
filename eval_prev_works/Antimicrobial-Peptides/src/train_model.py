import os
import pandas as pd
from src.load_data import load_df_from_dbs
from src.nn import conv_model, evaluate, evaluate_as_classifier, evaluate_model
from src.settings import MAX_SEQUENCE_LENGTH, character_to_index, CHARACTER_DICT, max_mic_buffer, MAX_MIC
from pyutils import Colors

from sklearn.model_selection import train_test_split
import numpy as np
import random
from Bio import SeqIO
import argparse
import matplotlib.pyplot as plt
from scipy import stats
from qmap.benchmark import QMAPBenchmark, DBAASPDataset
from qmap.toolkit import compute_maximum_identity

def get_bacterium_df(bacterium, df):
    bacterium_df = df.loc[(df.bacterium.str.contains(bacterium))].groupby(['sequence', 'bacterium'])
    return bacterium_df.mean().reset_index().dropna() # Mean duplicates


def sequence_to_vector(sequence):
    # One hot encoding
    vector = np.zeros([MAX_SEQUENCE_LENGTH, len(character_to_index) + 1])
    for i, character in enumerate(sequence[:MAX_SEQUENCE_LENGTH]):
        vector[i][character_to_index[character]] = 1
    return vector

def generate_random_sequence(min_length=5, max_length=MAX_SEQUENCE_LENGTH, fixed_length=None):
    if fixed_length:
        length = fixed_length
    else:
        length = random.choice(range(min_length, max_length))
    sequence = [random.choice(list(CHARACTER_DICT)) for _ in range(length)]
    return sequence

def add_random_negative_examples(vectors, labels, negatives_ratio):
    if negatives_ratio == 0:
        return vectors, labels
    num_negative_vectors = int(negatives_ratio * len(vectors))
    negative_vectors = np.array(
        [sequence_to_vector(generate_random_sequence()) for _ in range(num_negative_vectors)]
    ) 
    vectors = np.concatenate((vectors, negative_vectors))
    negative_labels = np.full(num_negative_vectors, MAX_MIC)
    labels = np.concatenate((labels, negative_labels))
    return vectors, labels

def load_uniprot_negatives(count):
    uniprot_file = 'data/Fasta_files/Uniprot_negatives.txt'
    fasta = SeqIO.parse(uniprot_file, 'fasta')
    fasta_sequences = [str(f.seq) for f in fasta]
    negatives = []
    for seq in fasta_sequences:
        if 'C' in seq:
            continue
        start = random.randint(0,len(seq)-MAX_SEQUENCE_LENGTH)
        negatives.append(seq[start:(start+MAX_SEQUENCE_LENGTH)])
        if len(negatives) >= count:
            return negatives
    return negatives

def uniprot_precision(model):
    negatives = load_uniprot_negatives(1000)
    vectors = []
    for seq in negatives:
        try:
            vectors.append(sequence_to_vector(seq))
        except KeyError:
            continue
    preds = model.predict(np.array(vectors))
    false_positives = len([p for p in preds if p < MAX_MIC - max_mic_buffer])
    return 1 - false_positives / len(negatives)


def train_model(bacterium, negatives_ratio=1, epochs=100):
    """
    Bacterium can be E. coli, P. aeruginosa, etc.
    When with_negatives is False, classification error will be 0
    and error on correctly classified/active only/all will be equal
    because all peptides in the dataset are active
    """
    DATA_PATH = 'data/'
    df = load_df_from_dbs(DATA_PATH)
    bacterium_df = get_bacterium_df(bacterium, df)
    print("Found %s sequences for %s" % (len(bacterium_df), bacterium))
    bacterium_df['vector'] = bacterium_df.sequence.apply(sequence_to_vector)

    x = np.array(list(bacterium_df.vector.values))
    y = bacterium_df.value.values
    x, y = add_random_negative_examples(x, y, negatives_ratio)

    train_x, test_x, train_y, test_y = train_test_split(x, y)

    model = conv_model()
    model.fit(train_x, train_y, epochs=epochs)
    print("Avg. MIC error (correctly classified, active only, all)")
    print(evaluate(model, test_x, test_y))
    print(evaluate_model(model, test_x, test_y))

    return model

def train_model_qmap(bacterium, negatives_ratio=1, epochs=100, train_mode=None):
    """
    Bacterium can be E. coli, P. aeruginosa, etc.
    When with_negatives is False, classification error will be 0
    and error on correctly classified/active only/all will be equal
    because all peptides in the dataset are active
    """
    DATA_PATH = 'data/'
    df = load_df_from_dbs(DATA_PATH)
    bacterium_df = get_bacterium_df(bacterium, df)
    # Filter out sequence larger than 100
    bacterium_df = bacterium_df.loc[bacterium_df['sequence'].str.len() < 100]
    print("Found %s sequences for %s" % (len(bacterium_df), bacterium))
    bacterium_df['vector'] = bacterium_df.sequence.apply(sequence_to_vector)

    if train_mode == 'rnd':
        x = np.array(list(bacterium_df.vector.values))
        y = bacterium_df.value.values  # log10(MIC_uM)
        sequences = bacterium_df['sequence'].tolist()

        rng = np.random.default_rng(42)
        perm = rng.permutation(len(x))
        split_point = int(0.8 * len(x))
        train_idx, test_idx = perm[:split_point], perm[split_point:]

        train_x, train_y = x[train_idx], y[train_idx]
        random.seed(42)
        train_x, train_y = add_random_negative_examples(train_x, train_y, negatives_ratio)

        test_sequences = [sequences[i] for i in test_idx]
        test_mic_uM = 10 ** y[test_idx]
        test_x = np.array([sequence_to_vector(seq) for seq in test_sequences])
        eval_benchmark = DBAASPDataset([
            {
                'id': i,
                'sequence': seq,
                'smiles': [],
                'nterminal': None,
                'cterminal': None,
                'bonds': [],
                'targets': {'Escherichia coli': (float(mic), float(mic), float(mic))},
                'hemolytic_hc50': None
            }
            for i, (seq, mic) in enumerate(zip(test_sequences, test_mic_uM))
        ])

        model = conv_model()
        model.fit(train_x, train_y, epochs=epochs)
        preds = model.predict(test_x)[:, 0]
        preds = [{'Escherichia coli': val.item()} for val in preds]
        results = eval_benchmark.compute_metrics(preds)["Escherichia coli"]
        print(f"{Colors.green}{results}{Colors.reset}")

        if not os.path.exists('results'):
            os.makedirs('results')
        pd.DataFrame([results.dict()]).to_csv('results/rnd_split.csv')
        print(results.md_col, end="")
        print(results.md_row, end="")

        # Identity scatter: max train-test identity vs absolute error
        train_sequences = [sequences[i] for i in train_idx]
        max_identity = compute_maximum_identity(train_sequences, test_sequences)

        pred_values = model.predict(test_x)[:, 0]
        test_targets = y[test_idx]
        abs_error = np.abs(pred_values - test_targets)

        pearson_r, pearson_p = stats.pearsonr(max_identity, abs_error)
        spearman_r, spearman_p = stats.spearmanr(max_identity, abs_error)

        slope, intercept, *_ = stats.linregress(max_identity, abs_error)
        x_line = np.linspace(max_identity.min(), max_identity.max(), 200)
        y_line = slope * x_line + intercept

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(max_identity, abs_error, alpha=0.4, s=15)
        ax.plot(x_line, y_line, color='tab:red', linewidth=1.5, label='Linear fit')
        ax.set_xlabel('Max sequence identity to nearest train sample')
        ax.set_ylabel('Absolute error (log₁₀ MIC)')
        ax.set_title('Identity vs prediction error — random split (AMP conv / E. coli)')
        ax.set_ylim(0, None)
        ax.text(0.03, 0.97,
                f"Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\nSpearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.legend()
        plt.tight_layout()
        plt.savefig('results/rnd_identity_vs_error.png', dpi=150)
        plt.show()
        return

    # Make the train set
    all_results = []
    all_results_high_eff = []
    for i in range(5):
        print(f'{Colors.orange}Running split {i}{Colors.reset}')
        benchmark = (QMAPBenchmark(i)
                         .with_bacterial_targets(["Escherichia coli"])
                         .with_canonical_only()
                         .with_l_aa_only()
                         .with_terminal_modification(False, False)
                         .with_length_range(None, 100)
                         )
        x = np.array(list(bacterium_df.vector.values))
        y = bacterium_df.value.values  # log10(MIC_uM)

        # Mask sequences too close to the test set
        sequences = bacterium_df['sequence'].tolist()
        mask = benchmark.get_train_mask(sequences)
        train_x_all = x[mask]
        train_y_all = y[mask]

        n_he = int((train_y_all < np.log10(10)).sum())
        if train_mode == 'he':
            he = train_y_all < np.log10(10)
            train_x = train_x_all[he]
            train_y = train_y_all[he]
        elif train_mode == 'me':
            me = (train_y_all >= np.log10(10)) & (train_y_all <= np.log10(100))
            train_x, train_y = train_x_all[me], train_y_all[me]
            if len(train_y) > n_he:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(train_y), n_he, replace=False)
                train_x, train_y = train_x[idx], train_y[idx]
        elif train_mode == 'le':
            le = train_y_all > np.log10(100)
            train_x, train_y = train_x_all[le], train_y_all[le]
            if len(train_y) > n_he:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(train_y), n_he, replace=False)
                train_x, train_y = train_x[idx], train_y[idx]
        else:
            train_x, train_y = train_x_all, train_y_all

        train_x, train_y = add_random_negative_examples(train_x, train_y, negatives_ratio)

        if train_mode == 'he':
            eval_benchmark = benchmark.with_efficiency_below(10.)
        elif train_mode == 'me':
            eval_benchmark = benchmark.filter(lambda s: any(10 <= t.consensus <= 100 for t in s.targets.values()))
        elif train_mode == 'le':
            eval_benchmark = benchmark.filter(lambda s: any(t.consensus > 100 for t in s.targets.values()))
        else:
            eval_benchmark = benchmark

        test_x, test_y = eval_benchmark.tabular(["sequence"])["sequence"].tolist(), np.log10(eval_benchmark.tabular(["Escherichia coli"])["Escherichia coli"].values)
        test_x = np.array([sequence_to_vector(seq) for seq in test_x])

        model = conv_model()
        model.fit(train_x, train_y, epochs=epochs)
        print(f"{Colors.green}Avg. MIC error (correctly classified, active only, all)")
        print(evaluate(model, test_x, test_y))
        preds = model.predict(test_x)[:, 0]
        preds = [{'Escherichia coli': val.item()} for val in preds]
        results = eval_benchmark.compute_metrics(preds)["Escherichia coli"]
        all_results.append(results)
        print(results)
        print(Colors.reset)

        if not train_mode:
            high_eff_benchmark = benchmark.with_efficiency_below(10.)
            test_x = np.array([sequence_to_vector(seq) for seq in high_eff_benchmark.tabular(["sequence"])["sequence"].tolist()])
            preds = model.predict(test_x)[:, 0]
            preds = [{'Escherichia coli': val.item()} for val in preds]
            all_results_high_eff.append(high_eff_benchmark.compute_metrics(preds)["Escherichia coli"])


    all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])

    # Export to pandas
    if not os.path.exists('results'):
        os.makedirs('results')
    if train_mode:
        all_result_table.to_csv(f'results/train_{train_mode}.csv')
    else:
        high_efficiency = pd.DataFrame([result.dict() for result in all_results_high_eff])
        all_result_table.to_csv('results/full.csv')
        high_efficiency.to_csv('results/high_efficiency.csv')

    print(all_results[0].md_col, end="")
    for results in all_results:
        print(results.md_row, end="")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bacterium', type=str, default='E. coli', help='Name of bacterium, in single quotes')
    parser.add_argument('--negatives', type=float, default=1, help='Ratio of negatives to positives')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    train_model(bacterium=args.bacterium, negatives_ratio=args.negatives, epochs=args.epochs)
