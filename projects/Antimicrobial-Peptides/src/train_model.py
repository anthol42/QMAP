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
from qmap.benchmark import QMAPBenchmark

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

def train_model_qmap(bacterium, negatives_ratio=1, epochs=100):
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

    # Make the train set
    all_results = []
    all_results_high_complexity = []
    all_results_low_complexity = []
    all_results_high_eff = []
    for i in range(5):
        for threshold in [55, 60]:
            print(f'{Colors.orange}Running split {i}{Colors.reset}')
            benchmark = QMAPBenchmark(i, threshold,
                                      species_subset=['Escherichia coli'],
                                      )
            x = np.array(list(bacterium_df.vector.values))
            y = bacterium_df.value.values

            # Mask sequences too close to the test set
            sequences = bacterium_df['sequence'].tolist()
            mask = benchmark.get_train_mask(sequences)
            train_x = x[mask]
            train_y = y[mask]
            train_x, train_y = add_random_negative_examples(train_x, train_y, negatives_ratio)

            test_x, test_y = benchmark.inputs, benchmark.targets
            test_x = np.array([sequence_to_vector(seq) for seq in test_x])

            model = conv_model()
            model.fit(train_x, train_y, epochs=epochs)
            print(f"{Colors.green}Avg. MIC error (correctly classified, active only, all)")
            print(evaluate(model, test_x, test_y))
            preds = model.predict(test_x)
            results = benchmark.compute_metrics(preds)
            all_results.append(results)
            print(results)
            print(Colors.reset)

            high_comp_benchmark = benchmark.high_complexity
            test_x = np.array([sequence_to_vector(seq) for seq in high_comp_benchmark.inputs])
            preds = model.predict(test_x)
            all_results_high_complexity.append(high_comp_benchmark.compute_metrics(preds))

            low_comp_benchmark = benchmark.low_complexity
            test_x = np.array([sequence_to_vector(seq) for seq in low_comp_benchmark.inputs])
            preds = model.predict(test_x)
            all_results_low_complexity.append(low_comp_benchmark.compute_metrics(preds))

            high_eff_benchmark = benchmark.high_efficiency
            test_x = np.array([sequence_to_vector(seq) for seq in high_eff_benchmark.inputs])
            preds = model.predict(test_x)
            all_results_high_eff.append(high_eff_benchmark.compute_metrics(preds))


    all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])
    high_complexity = pd.DataFrame([result.dict() for result in all_results_high_complexity])
    low_complexity = pd.DataFrame([result.dict() for result in all_results_low_complexity])
    high_efficiency = pd.DataFrame([result.dict() for result in all_results_high_eff])

    # Export to pandas
    if not os.path.exists('results'):
        os.makedirs('results')
    all_result_table.to_csv('results/full.csv')
    high_complexity.to_csv('results/high_complexity.csv')
    low_complexity.to_csv('results/low_complexity.csv')
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
