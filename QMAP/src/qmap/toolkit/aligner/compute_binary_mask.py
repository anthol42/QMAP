import pwiden_engine as pe
from typing import List, Literal, Optional
import numpy as np

def compute_binary_mask(train_sequences: List[str],
                            test_sequences: List[str],
                            threshold: float = 0.6,
                            matrix: str = "blosum45",
                            gap_open: int = 5,
                            gap_extension: int = 1,
                            use_cache: bool = True,
                            show_progress: bool = True,
                            num_threads: Optional[int] = None,
                            ) -> np.ndarray:
    """

    :param train_sequences: List of training sequences
    :param test_sequences: List of test sequences
    :param threshold: Minimum similarity threshold to save the edge.
    :param matrix: Substitution matrix name (default: "blosum45")
Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
Also: pam{10-500} in steps of 10
    :param gap_open: Gap opening penalty
    :param gap_extension: Gap extension penalty
    :param use_cache: Whether to use caching (default: True)
    :param show_progress: Whether to show progress bar
    :param num_threads: Number of threads to use for parallel computation (default: None = all available cores)
    :return: A 1D numpy boolean array of length n_train. True indicates the training sequence
    should be removed (has identity >= threshold with at least one test sequence).
    """
    return pe.compute_binary_mask(
        train_sequences,
        test_sequences,
        threshold,
        matrix,
        gap_open,
        gap_extension,
        use_cache,
        show_progress,
        num_threads
    )
