import numpy as np
from ..aligner import compute_global_identity
from tqdm import tqdm

class Identity:
    """
    Contains the identity matrix between all sequences in the dataset. This allows to quickly simulate
    the calculation of the identity between two sequences without having to compute it on the fly.
    """
    def __init__(self, sequences: list[str], *args, **kwargs):
        """
        :param sequences: The list of sequences of the dataset
        :param args: positional arguments for qmap.toolkit.aligner.compute_global_identity
        :param kwargs: keyword arguments for qmap.toolkit.aligner.compute_global_identity
        """
        self.dm = compute_global_identity(sequences, *args, **kwargs)
        self.dataset = {i: seq for i, seq in enumerate(sequences)}

    def align_by_id(self, idx1: int, idx2: int) -> float:
        """
        Returns the identity between two sequences given their IDs.
        """
        return self.dm[idx1, idx2].item()




def compute_maximum_identity(train_ids: list[int], test_ids: list[int], identity_calculator: Identity) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the identity statistics between the training and test sets.
    :param train_ids: List of sequence IDs in the training set.
    :param test_ids: List of sequence IDs in the test set.
    :param identity_calculator: Identity calculator.
    :return: An array containing the highest identity between each test samples and a mask of the true independent train samples.
    """
    identities = np.full((len(test_ids), len(train_ids)), np.nan)
    true_train_set_mask = np.ones(len(train_ids), dtype=bool)
    for i, test_id in enumerate(tqdm(test_ids)):
        for j, train_id in enumerate(train_ids):
            identity = identity_calculator.align_by_id(train_id, test_id)
            if identity > 0.5:
                true_train_set_mask[j] = False
            identities[i, j] = identity
    return identities, true_train_set_mask