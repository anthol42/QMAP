import pandas as pd

from qmap.toolkit import aligner
from qmap.toolkit.clustering import build_graph, leiden_community_detection
from qmap.toolkit import split
from generate_esm_embeddings import generate_esm2_embeddings
import pickle
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

def sequence_entropy(sequence: str) -> float:
    """
    Calculate the Shannon entropy of a sequence.
    :param sequence: The sequence to calculate the entropy for
    :return: The Shannon entropy of the sequence
    """
    from collections import Counter
    from math import log2

    counts = Counter(sequence)
    total = len(sequence)
    probabilities = [count / total for count in counts.values()]
    return -sum(p * log2(p) for p in probabilities if p > 0)

def read_fasta(file_path):
    """
    Reads a FASTA file and returns a list of tuples containing sequence IDs and sequences.
    :param file_path: The path to the FASTA file.
    :return:
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sequences = {}
    for line in lines:
        if line.startswith('>'):
            id_ = int(line[1:].strip().replace("seq_", ""))
        else:
            sequence = line.strip()
            sequences[id_] = sequence
    return sequences

if __name__ == "__main__":
    dsequences = read_fasta("../data/build/dataset.fasta")
    dsequences = {id_: seq for id_, seq in dsequences.items() if len(seq) <= 100}
    encoder = aligner.Encoder(force_cpu=True)
    ids = list(dsequences.keys())
    sequences = [dsequences[id_] for id_ in ids]
    embeddings = generate_esm2_embeddings(sequences, device="mps", batch_size=512)
    results = []
    for annot_method in ["random", "prob", "max"]:
        results.append([])
        for split_met in ["prob", "random", "max"]:
            train_sequences, test_sequences, neg, pos = split.train_test_split(sequences, embeddings, test_size=0.45, method=annot_method, threshold=0.6)
            all_sequences = train_sequences + test_sequences
            X = np.concatenate((neg, pos), axis=0)
            y = np.array([0] * len(neg) + [1] * len(pos))

            # Split dataset into training and test sets
            seq_train, seq_test, X_train, X_test, y_train, y_test = split.train_test_split(all_sequences, X, y, test_size=0.2, shuffle=True, method=split_met, threshold=0.6)
            classifier = LogisticRegression(max_iter=10000)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            print(np.unique(y_train, return_counts=True))
            print(f"{annot_method} - {split_met}: Test accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
            results[-1].append(balanced_accuracy_score(y_test, y_pred))
    results = pd.DataFrame(results, columns=["prob", "random", "max"])
    results["annot_method"] = ["random", "prob", "max"]
    results.to_csv("tmp_results.csv", index=False)
