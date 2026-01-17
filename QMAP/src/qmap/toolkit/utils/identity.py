import os
import hashlib
import numpy as np
import numpy.typing as npt
from typing import Optional
from ..aligner import compute_global_identity
from tqdm import tqdm
import pwiden_engine as pe


def _hash_sequences(sequences: list[str]) -> str:
    """
    Compute a stable, content-based hash for a list of sequences.
    Order-sensitive.
    """
    h = hashlib.sha256()
    for seq in sequences:
        h.update(seq.encode("utf-8"))
        h.update(b"\0")  # separator to avoid accidental collisions
    return h.hexdigest()


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
        cache_dir = ".cache"
        os.makedirs(cache_dir, exist_ok=True)

        cache_id = _hash_sequences(sequences)
        cache_path = os.path.join(cache_dir, f"{cache_id}.npy")

        if os.path.exists(cache_path):
            self.dm = np.load(cache_path)
        else:
            self.dm = compute_global_identity(sequences, *args, **kwargs)
            np.save(cache_path, self.dm)

        self.dataset = {i: seq for i, seq in enumerate(sequences)}

    def align_by_id(self, idx1: int, idx2: int) -> float:
        """
        Returns the identity between two sequences given their IDs.
        """
        return self.dm[idx1, idx2].item()



def compute_maximum_identity(train_sequences: list[str],
                             test_sequences: list[str],
                            matrix: str = "blosum45",
                            gap_open: int = 5,
                            gap_extension: int = 1,
                            use_cache: bool = True,
                            show_progress: bool = True,
                            num_threads: Optional[int] = None,
                             ) -> npt.NDArray[np.float32]:
    """
    Use the pwiden engine to quickly compute the maximum identity metri distribution between the training and test sets.
    :param train_sequences: The training sequences.
    :param test_sequences: The test sequences.
    :param matrix: Substitution matrix name (default: "blosum45")
Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
Also: pam{10-500} in steps of 10
    :param gap_open: Gap opening penalty
    :param gap_extension: Gap extension penalty
    :param use_cache: Whether to use caching (default: True)
    :param show_progress: Whether to show progress bar
    :param num_threads: Number of threads to use for parallel computation (default: None = all available cores)
    :return: The maximum identity vector, the same length as the test set.
    """
    return pe.compute_maximum_identity(
        train_sequences,
        test_sequences,
        matrix,
        gap_open,
        gap_extension,
        use_cache,
        show_progress,
        num_threads
    )
