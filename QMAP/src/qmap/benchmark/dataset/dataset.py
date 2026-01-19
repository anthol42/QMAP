from scipy.stats import spearmanr, kendalltau, pearsonr
from typing import Callable, Optional, Literal, Union, List
from huggingface_hub import hf_hub_download
from collections import defaultdict
import pandas as pd
import numpy as np
import json
import math

from .bond import Bond
from .sample import Sample
from .target import Target
import pwiden_engine as pe
from .metrics import r2_score
from .hemolytic import HemolyticActivity
from .QMAP_metrics import QMAPRegressionMetrics
from .filters import (filter_bacteria, filter_efficiency_below, filter_hc50, filter_canonical_only,
                      filter_common_only, filter_l_aa_only, filter_terminal_modification, filter_bond)


def _format_sample_line(sample: Sample) -> str:
    free_str = 'Free'
    return f'{str(sample.id).ljust(10, " ")} {sample.sequence[:40].ljust(40, " ")} {(sample.nterminal or free_str).ljust(12)} {(sample.cterminal or free_str).ljust(12)} {str(len(sample.targets)).ljust(10)} {sample.hc50.consensus if sample.hc50 else np.nan:.2f}'


class DBAASPDataset:
    """
    Class representing a the DBAASP dataset used to build the benchmark.

    ## Example:
    dataset = DBAASPDataset(...)

    dataset_filtered = (dataset
               .with_bacterial_targets(["Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"])
               .with_efficiency_below(10.0)
               .with_common_only()
               .with_l_aa_only()
               .with_terminal_modification(None, None)
               )
    print(dataset)

    # You can also extend the dataset with your own samples:
    dataset_extended = dataset_filtered + DBAASPDataset(your_own_data)
    # Or with the extend method
    dataset_extended = dataset_filtered.extend(DBAASPDataset(your_own_data))

    # You can also index the dataset like a list, or a numpy array:
    first_sample = dataset[0]
    some_samples = dataset[1:10]
    boolean_indexed_samples = dataset[np.array([True, False, True, False, ...])]
    only_certain_samples = dataset[[0, 2, 5, 7]]
    """
    def __init__(self, data: Optional[list[dict]] = None):
        """
        When constructing the dataset from raw data, the data should be a list of dictionaries with the following format:
        {
            'id': int,
            'sequence': str,
            'smiles': list[str],
            'nterminal': Optional['ACT'],
            'cterminal': Optional['AMD'],
            'bonds': list[tuple[int, int, 'DSB' | 'AMD']],
            'targets': dict[str, tuple[float, float, float]],
            'hemolytic_hc50': Optional[tuple[float, float, float]]
        }

        If data is None, it loads the dataset from HuggingFace Hub.
        """
        if data is None:
            path = hf_hub_download(
                repo_id="anthol42/qmap_benchmark_2025",
                filename="dbaasp.json",
                repo_type="dataset"
            )

            with open(path, 'r') as f:
                data = json.load(f)
        self.samples = [Sample.FromDict(sample_data) for sample_data in data]

    @property
    def sequences(self) -> list[str]:
        """
        Return the list of sequences in the dataset.
        """
        return [sample.sequence for sample in self.samples]

    @property
    def ids(self) -> list[int]:
        """
        Return the list of DBAASP IDs in the dataset.
        """
        return [sample.id for sample in self.samples]

    @property
    def smiles(self) -> list[list[str]]:
        """
        Return the list of SMILES strings in the dataset.
        """
        return [sample.smiles for sample in self.samples]

    @property
    def nterminals(self) -> list[Optional[str]]:
        """
        Return the list of N-terminal modifications in the dataset.
        """
        return [sample.nterminal for sample in self.samples]

    @property
    def cterminals(self) -> list[Optional[str]]:
        """
        Return the list of C-terminal modifications in the dataset.
        """
        return [sample.cterminal for sample in self.samples]

    @property
    def bonds(self) -> list[Bond]:
        """
        Return the list of bonds in the dataset.
        """
        return [sample.bonds for sample in self.samples]

    @property
    def targets(self) -> list[list[Target]]:
        """
        Return the list of targets in the dataset.
        """
        return [list(sample.targets.values) for sample in self.samples]

    @property
    def hc50s(self) -> list[Optional[HemolyticActivity]]:
        """
        Return the list of hemolytic activities (HC50) in the dataset.
        """
        return [sample.hc50 for sample in self.samples]

    @classmethod
    def FromSamples(cls, samples: list[Sample]) -> "DBAASPDataset":
        """
        Load a dataset from a list of Sample objects, instead of from raw data.
        """
        self = DBAASPDataset([])
        self.samples = samples
        return self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: Union[int, slice, np.ndarray, list[bool], list[int]]):
        if isinstance(idx, np.ndarray):
            if idx.dtype == bool:
                return DBAASPDataset.FromSamples([sample for sample, keep in zip(self.samples, idx) if keep])
            else:
                return DBAASPDataset.FromSamples([self.samples[i] for i in idx.tolist()])
        elif isinstance(idx, list) and all(isinstance(i, bool) for i in idx):
            return DBAASPDataset.FromSamples([sample for sample, keep in zip(self.samples, idx) if keep])
        elif isinstance(idx, list) and all(isinstance(i, int) for i in idx):
            return DBAASPDataset.FromSamples([self.samples[i] for i in idx])
        elif isinstance(idx, int):
            return self.samples[idx]
        elif isinstance(idx, slice):
            return self.samples[idx]
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self):
        return f"DBAASPDataset({self.samples})"

    def __str__(self):
        columns = f'{"DBAASP ID".ljust(10, " ")} {"Sequence".ljust(40, " ")} {"N Terminal".ljust(12)} {"C Terminal".ljust(12)} {"# Targets".ljust(10, " ")} {"HC50".ljust(6, " ")}'
        if len(self) > 10:
            first_four = [_format_sample_line(sample) for sample in self.samples[:4]]
            middle = len(columns) // 2 * " " + "..."
            last_four = [_format_sample_line(sample) for sample in self.samples[-4:]]
            return columns + "\n"+ "\n".join(first_four) + "\n" + middle + "\n" + "\n".join(last_four) + f"\n\nTotal of {len(self)} samples"
        else:
            lines = [_format_sample_line(sample) for sample in self.samples]
            return columns + "\n" + "\n".join(lines) + f"\n\nTotal of {len(self)} samples"

    def __add__(self, other):
        return self.extend(other)

    def compute_metrics(self, predictions: list[dict[str, float]], log: bool = True, mean_metrics: bool = True) -> dict[str, QMAPRegressionMetrics]:
        """
        Compute metrics on the dataset given the predictions. The predictions must be in a specific format:

        A list of dictionaries, where each dictionary corresponds to the prediction for a sample, and each key in the
        dictionary is associated to a property name. These keys can be any bacterial target name, or hc50 for
        hemolytic activity. If a label exist for the sample and the property, it will be used to compute the metrics.
        Otherwise, the prediction is ignored. Please, make sure that all the keys do exist, otherwise the function will
        fail silently by returning nan metrics for this property.

        > **IMPORTANT:**
        > The order of the predictions must match the order of the samples in the dataset.
        > Thus, the length of the predictions list must be equal to the length of the dataset.

        :param predictions: The predictions to evaluate.
        :param log: If True, apply a log10 on the targets before computing the metrics. This means that the prediction
        are made in the log form. This is recommended.
        :param mean_metrics: If True, a key will be added to the output containing the mean metrics across all
        properties. The key is named 'mean'.
        :return: A dictionary of QMAPMetrics objects, one for each property predicted. The key is the property name,
        and the value the set of metrics.
        """
        if len(predictions) != len(self):
            raise ValueError("The length of the predictions must be equal to the length of the dataset.")

        all_preds= defaultdict(list)
        all_targets = defaultdict(list)
        for sample, pred in zip(self.samples, predictions):
            if not isinstance(pred, dict):
                raise ValueError("Each prediction must be a dictionary mapping property names to predicted values.")
            # Gather predictions and true values for each property
            for key, value in pred.items():
                if key == "hc50" and not math.isnan(sample.hc50.consensus):
                    all_targets["hc50"].append(sample.hc50.consensus)
                    all_preds["hc50"].append(value)
                elif key in sample.targets:
                    all_targets[key].append(sample.targets[key].consensus)
                    all_preds[key].append(value)

        assert all_preds.keys() == all_targets.keys(), "All keys should be the same"
        for key in all_preds.keys():
            assert len(all_preds[key]) == len(all_targets[key]), f"All lengths should be the same, found different lengths for key {key} ({len(all_preds[key])} != {len(all_targets[key])})"

        all_metrics = {}
        for key in all_preds.keys():
            metric = self._regression_metrics(np.array(all_preds[key]), np.array(all_targets[key]), property_name=key, log=log)
            all_metrics[key] = metric

        if mean_metrics:
            # First, get the total number of measurements
            total_measurements = sum(metric.n for metric in all_metrics.values())
            # Then, compute the weighted mean for each metric
            mean_rmse = sum(metric.rmse * metric.n for metric in all_metrics.values()) / total_measurements
            mean_mse = sum(metric.mse * metric.n for metric in all_metrics.values()) / total_measurements
            mean_mae = sum(metric.mae * metric.n for metric in all_metrics.values()) / total_measurements
            mean_r2 = sum(metric.r2 * metric.n for metric in all_metrics.values()) / total_measurements
            mean_spearman = sum(metric.spearman * metric.n for metric in all_metrics.values()) / total_measurements
            mean_kendalls_tau = sum(metric.kendalls_tau * metric.n for metric in all_metrics.values()) / total_measurements
            mean_pearson = sum(metric.pearson * metric.n for metric in all_metrics.values()) / total_measurements
            mean_metric = QMAPRegressionMetrics(
                property_name="mean",
                n=total_measurements,
                rmse=mean_rmse,
                mse=mean_mse,
                mae=mean_mae,
                r2=mean_r2,
                spearman=mean_spearman,
                kendalls_tau=mean_kendalls_tau,
                pearson=mean_pearson
            )
            all_metrics["mean"] = mean_metric

        return all_metrics

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
        return ~pe.compute_binary_mask(sequences, self.sequences,
                                   threshold=threshold,
                                   matrix=matrix,
                                   gap_open=gap_open,
                                   gap_extension=gap_extension,
                                   use_cache=use_cache,
                                   show_progress=show_progress,
                                   num_threads=num_threads
                                   )


    def extend(self, other: 'DBAASPDataset') -> 'DBAASPDataset':
        """
        Extend the dataset with another DBAASPDataset.
        """
        extended_samples = self.samples + other.samples
        return DBAASPDataset.FromSamples(extended_samples)

    def tabular(self, columns: list[str]) -> pd.DataFrame:
        """
        Convert the sample to tabular data. Only these fields are supported:
        - id: DBAASP ID
        - sequence: noncanonical: O is Ornithine, B is DAB
        - smiles: Note that only the first SMILES string is used
        - nterminal: None or ACT
        - cterminal: None or AMD
        - targets <target_name>: Note that only the consensus value is used
        - hc50: Note that only the consensus value is used

        ## Example:
        ```
        df = dataset.tabular(["id", "sequence", "nterminal", "cterminal", "hc50", "Escherichia coli"])
        print(df)
        ```
        """
        rows = [sample.tabular(columns) for sample in self.samples]
        return pd.DataFrame(rows, columns=columns)


    def filter(self, mapper: Callable[[Sample], bool]) -> 'DBAASPDataset':
        """
        Filter the dataset using a mapper function that takes a Sample and returns a boolean.
        A sample is kept if the mapper returns True, and drop otherwise.
        """
        filtered_samples = [sample for sample in self.samples if mapper(sample)]
        return DBAASPDataset.FromSamples(filtered_samples)

    def with_bacterial_targets(self, allowed: list[str]) -> 'DBAASPDataset':
        """
        Keep only samples that have at least one bacterial target in the allowed list.
        """
        return self.filter(filter_bacteria(allowed))

    def with_efficiency_below(self, threshold: float) -> 'DBAASPDataset':
        """
        Keep only samples that have at least one bacterial target with efficiency below the given threshold (in µM).
        """
        return self.filter(filter_efficiency_below(threshold))

    def with_hc50(self) -> 'DBAASPDataset':
        """
        Keep only samples that have hemolytic activity (HC50) reported.
        """
        return self.filter(filter_hc50())

    def with_canonical_only(self) -> 'DBAASPDataset':
        """
        Keep only samples that have only canonical amino acids in their sequence.
        Sequences with non-canonical amino acids represented by the letters O (Ornithine) and B (DAB) or X are
        filtered out.
        """
        return self.filter(filter_canonical_only())

    def with_common_only(self) -> 'DBAASPDataset':
        """
        Keep only samples that have only common amino acids in their sequence: canonical amino acids and
        Ornithin (O) and DAB (B).
        """
        return self.filter(filter_common_only())

    def with_l_aa_only(self) -> 'DBAASPDataset':
        """
        Keep only samples that have only L-amino acids in their sequence.
        D-amino acids are represented by lowercase letters.
        """
        return self.filter(filter_l_aa_only())

    def with_terminal_modification(self, nterminal: Optional[bool], cterminal: Optional[bool]) -> 'DBAASPDataset':
        """
        Keep only samples that have the specified terminal modifications.
        :param nterminal: N-terminal modification to filter by. Use None for free N-terminus, or 'ACT' for acetylation. Use True to keep only sequence having 'ACT' modification, or False to keep only sequences with free N-terminus.
        :param cterminal: C-terminal modification to filter by. Use None for free C-terminus, or 'AMD' for amidation. Use True to keep only sequence having 'AMD' modification, or False to keep only sequences with free C-terminus.
        """
        return self.filter(filter_terminal_modification(nterminal, cterminal))

    def with_bond(self, bond_type: list[Literal['DSB', 'AMD', None]]) -> 'DBAASPDataset':
        """
        Keep only samples that have bonds within the specified bond types. None means no bond is allowed.
        For example, to keep only samples with disulfide bonds, use bond_type=['DSB']. To keep only samples with no
        bonds, use bond_type=[None].
        """
        return self.filter(filter_bond(bond_type))


    def _regression_metrics(self, predictions: np.ndarray, targets: np.ndarray, property_name: str, log: bool = True) -> QMAPRegressionMetrics:
        """
        Compute the QMAP metrics given the predictions of the model. The metrics computed are:
        - RMSE
        - MSE
        - MAE
        - R2
        - Spearman correlation
        - Kendall's tau
        - Pearson correlation

        :param predictions: The predictions to evaluate. It should have the same length and order as this dataset.
        :param targets: The targets to evaluate. It should have the same length and order as this dataset.
        :param property_name: The name of the property to compute the metrics for.
        :param log: If true, apply a log10 on the targets.
        :return: A QMAPRegressionMetrics object containing all the metrics.
        """
        targets = np.log10(targets) if log else targets
        mse = np.mean((targets- predictions) ** 2)
        mae = np.mean(np.abs(targets - predictions))
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        spearman = spearmanr(targets, predictions).statistic
        kendalls_tau = kendalltau(targets, predictions).statistic
        pearson = pearsonr(targets, predictions).statistic

        n = len(predictions)
        return QMAPRegressionMetrics(property_name=property_name,
                            n=n,
                           rmse=rmse,
                           mse=mse,
                           mae=mae,
                           r2=r2,
                           spearman=spearman,
                           kendalls_tau=kendalls_tau,
                           pearson=pearson)