import numpy as np
from typing import List, Optional
import json
from huggingface_hub import hf_hub_download
from .dataset import DBAASPDataset

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


