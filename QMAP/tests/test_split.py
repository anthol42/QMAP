import pytest
from qmap.toolkit import split
import pandas as pd


def test_random_cluster_split():
    clusters = pd.DataFrame({
        'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'community': [0, 0, 0, 0, 1, 1, 2, 3]
    })
    train_ids, test_ids = split.random_cluster_split(clusters, test_ratio=0.25)

    # Check that the same id is not in both train and test sets
    assert not set(train_ids).intersection(set(test_ids)), "Train and test sets should not overlap."
    # Check that the total number of ids is correct
    assert len(train_ids) + len(test_ids) == len(clusters), "Total number of ids in train and test sets should equal the number of clusters."

def test_filter_out(sequences):
    # First are train then test
    train = sequences[:6]
    test = sequences[6:]
    test.append("LXXVEG") # Add a sequence similar to one in train
    train = split.filter_out(train, ref_sequences=test)
    assert train == ['DSHAKRHHGYKRKFHEKHHSHRGY', 'KVVVKWVVKVVK', 'YVLLKRKRLIFI', 'VVVVVV', 'VRNHVTCRINRGFCVPIRCPGRTRQIGTCFGPRIKCCRSW']


    train = sequences[:6]
    ids = [f"id_{i}" for i in range(len(train))]
    train, ids = split.filter_out(train, ids, ref_sequences=test)
    assert ids == ['id_0', 'id_1', 'id_2', 'id_3', 'id_4']

def test_train_test_split_seq_only(sequences):
    # Test the shuffling
    sequences.append("LXXVEG")
    ids = [f"id_{i}" for i in range(len(sequences))]
    idsmapping = dict(zip(ids, sequences))
    train_sequences, test_sequences, train_ids, test_ids = split.train_test_split(sequences, ids, test_size=0.4, shuffle=True)

    # First, check that there are no overlaps
    assert not set(train_sequences).intersection(set(test_sequences)), "Train and test sets should not overlap."

    # Next, check that no sequences were dropped (post filtering is False by default)
    assert len(train_sequences) + len(test_sequences) == len(sequences), "Total number of sequences in train and test sets should equal the number of sequences."

    # Finally, check that the ids match the sequences
    for seq, id_ in zip(train_sequences, train_ids):
        assert seq == idsmapping[id_], f"Sequence {seq} does not match its ID {id_} in the training set."
    for seq, id_ in zip(test_sequences, test_ids):
        assert seq == idsmapping[id_], f"Sequence {seq} does not match its ID {id_} in the test set."


def test_train_test_split(sequences):
    sequences.append("LXXVEG")
    ids = [f"id_{i}" for i in range(len(sequences))]
    idsmapping = dict(zip(ids, sequences))

    train_sequences, test_sequences, train_ids, test_ids = split.train_test_split(
        sequences,
        ids,
        test_size=0.2,
        post_filtering=True)
    assert len(train_sequences) == len(train_ids), "Train and test sets should not overlap."

    # Check that the order of sequences and ids is the same
    assert [seq == idsmapping[id_] for seq, id_ in zip(train_sequences, train_ids)], "Sequences and IDs in the training set do not match."