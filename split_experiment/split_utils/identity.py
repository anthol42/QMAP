import numpy as np
from .functional import read_fasta
from pyutils import progress

class Identity:
    """
    Contains the identity matrix between all sequences in the dataset. This allows to quickly simulate
    the calculation of the identity between two sequences without having to compute it on the fly.
    """
    def __init__(self, path: str = ".cache", dataset_path: str = "../data/build/dataset.fasta"):
        self.dm = np.load(f"{path}/identity_matrix.npy")
        with open(f"{path}/row_ids.txt", 'r') as f:
            self.dm_ids = np.array([int(id_) for id_ in f.readlines()])

        dataset = read_fasta(dataset_path)
        self.dataset = {int(id_): seq for id_, seq in dataset}

    def align_by_id(self, id1: int, id2: int) -> float:
        """
        Returns the identity between two sequences given their IDs.
        """
        if id1 not in self.dataset:
            raise ValueError(f"IDs {id1}not found in the dataset.")
        if id2 not in self.dataset:
            raise ValueError(f"IDs {id2} not found in the dataset.")

        i = np.where(self.dm_ids == id1)[0][0]
        j = np.where(self.dm_ids == id2)[0][0]

        return self.dm[i, j]



def compute_identity_stats(train_ids, test_ids, identity_calculator):
    """
    Computes the identity statistics between the training and test sets.
    :param train_ids: List of sequence IDs in the training set.
    :param test_ids: List of sequence IDs in the test set.
    :param identity_calculator: Identity calculator.
    :return: An array containing the highest identity between each test samples and all train samples.
    """
    identities = np.full((len(test_ids), len(train_ids)), np.nan)
    true_train_set = np.ones(len(train_ids), dtype=bool)
    for i, test_id in enumerate(progress(test_ids)):
        for j, train_id in enumerate(train_ids):
            identity = identity_calculator.align_by_id(train_id, test_id)
            if identity > 0.5:
                true_train_set[j] = False
            identities[i, j] = identity
    return identities, true_train_set