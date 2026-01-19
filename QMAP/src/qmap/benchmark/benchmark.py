import numpy as np
from typing import List, Optional
import json
from huggingface_hub import hf_hub_download
from .dataset import DBAASPDataset
from ..toolkit import compute_binary_mask

class QMAPBenchmark(DBAASPDataset):
    """
    Class representing the QMAP benchmark testing dataset.

    Once you have the predictions of your model, you can call the `compute_metrics` method to compute the metrics of
    your model on the given test set. You can also evaluate the performances of your model on subsets of the benchmark
    by leveraging the filtering methods.

    To use the benchmark, you must select the split (from 0 to 4). It is highly
    recommended to test your model on all splits to get a better estimate of its real-world performance and to
    accurately compare it with other models. To do so, you must use the same hyperparameters, but change the training
    and validation dataset. For each split, use the `get_train_mask` method to get a mask indicating which sequences
    can be used in the training set and validation set. This mask will be True for sequences that are allowed in the
    training set and False for sequences that are too similar to a sequence in the test set. Train your model on the
    subset of your training dataset where the mask is True and evaluate it on the benchmark dataset. Do this for all
    splits. See the example section for more details.
    """
    def __init__(self, split: int = 0):
        if split not in [0, 1, 2, 3, 4]:
            raise ValueError("split must be one of 0, 1, 2, 3, or 4.")
        self.split = split

        path = hf_hub_download(
            repo_id="anthol42/qmap_benchmark_2025",
            filename=f"benchmark_split_{split}.json",
            repo_type="dataset"
        )
        with open(path, 'r') as f:
            data = json.load(f)
        super().__init__(data)


    def get_train_mask(self, sequences: List[str],
                       threshold: float = 0.6,
                       matrix: str = "blosum45",
                       gap_open: int = 5,
                       gap_extension: int = 1,
                       use_cache: bool = True,
                       show_progress: bool = True,
                       num_threads: Optional[int] = None,
                       ) -> np.ndarray:
        """
        Returns a mask indicating which sequences can be in the training set because they are not too similar to any
        other sequence in the test set. It returns a boolean mask where True means that the sequence is allowed in the
        training / validation set and False means that the sequence is too similar to a sequence in the test set and
        must be excluded.
        :param sequences: The training sequences to check against the benchmark test set.
        :param threshold: Minimum similarity threshold to save the edge.
        :param matrix: Substitution matrix name (default: "blosum45")
        Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
        Also: pam{10-500} in steps of 10
        :param gap_open: Gap opening penalty
        :param gap_extension: Gap extension penalty
        :param use_cache: Whether to use caching (default: True)
        :param show_progress: Whether to show progress bar
        :param num_threads: Number of threads to use for parallel computation (default: None = all available cores)
        :return: A binary mask where True means that the sequence is allowed in the training set and False means that the
        sequence is too similar to a sequence in the test set and must be excluded.
        """
        return compute_binary_mask(sequences, self.sequences,
                                   threshold=threshold,
                                   matrix=matrix,
                                   gap_open=gap_open,
                                   gap_extension=gap_extension,
                                   use_cache=use_cache,
                                   show_progress=show_progress,
                                   num_threads=num_threads
                                   )

