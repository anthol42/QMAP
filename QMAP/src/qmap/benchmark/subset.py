from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Literal
import numpy as np
from .QMAP_metrics import QMAPMetrics, r2_score
from scipy.stats import spearmanr, kendalltau, pearsonr

class BenchmarkSubset(Dataset):
    def __init__(self, split: int, threshold: Literal[55, 60], sequences: List[str], species: Optional[List[str]], targets: List[float], c_termini: List[str],
                     n_termini: List[str], unusual_aa: List[dict[int, str]], max_targets: List[float], min_targets: List[float],
                     *,
                     modified_termini: bool = False,
                     allow_unusual_aa: bool = False,
                     specie_as_input: bool = False,
                 ):
        self.split = split
        self.threshold = threshold / 100

        self.modified_termini = modified_termini
        self.allow_unusual_aa = allow_unusual_aa
        self.specie_as_input = specie_as_input

        self.sequences = sequences
        self.species = species
        self._targets = targets
        self.c_termini = c_termini
        self.n_termini = n_termini
        self.unusual_aa = unusual_aa
        self.max_targets = max_targets
        self.min_targets = min_targets


    @property
    def inputs(self) -> Tuple[List[str], ...]:
        """
        Depending on the initialization parameters, this property will return different values. It will return:
        - The sequence (str) always
        - The specie (str) if specie_as_input is True
        - The N terminus (str) if modified_termini is True
        - The C terminus (str) if modified_termini is True
        - The unusual amino acids (dict) if unusual_aa is True
        :return: All the inputs for the model.
        """
        out = [self.sequences]
        if self.specie_as_input:
            out.append(self.species)
        if self.modified_termini:
            out.append(self.n_termini)
            out.append(self.c_termini)
        if self.allow_unusual_aa:
            out.append(self.unusual_aa)
        return tuple(out) if len(out) > 1 else out[0]

    @property
    def targets(self) -> np.ndarray:
        return np.array(self._targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Tuple:
        """
        Depending on the initialization parameters, this method will return different values. It will return:
        - The sequence (str) always
        - The specie (str) if specie_as_input is True
        - The N terminus (str) if modified_termini is True
        - The C terminus (str) if modified_termini is True
        - The unusual amino acids (dict) if unusual_aa is True
        - The target (float or np.ndarray) always

        :param idx: The index of the sample to retrieve.
        :return: The inputs for a single sample and the target.
        """
        sequence = self.sequences[idx]
        out = [sequence]
        if self.specie_as_input:
            out.append(self.species[idx])
        if self.modified_termini:
            out.append(self.n_termini[idx])
            out.append(self.c_termini[idx])
        if self.allow_unusual_aa:
            out.append(self.unusual_aa[idx])

        target = self.targets[idx]
        return *out, target

    def accuracy(self, predictions: np.ndarray) -> float:
        """
        Compute the accuracy of the predictions. A good prediction is one that is within the MIC range if a range is
        provided. This method only work with MIC datasets.
        :param predictions: The predictions to evaluate. It should have the same length and order as this dataset.
        :return: The accuracy of the predictions.
        """
        mins = np.array(self.min_targets).reshape(-1)
        maxs = np.array(self.max_targets).reshape(-1)
        preds = predictions.reshape(-1)

        good = np.logical_and(preds >= mins, preds <= maxs)[mins != maxs]
        # Ignore nans
        good = good[~np.logical_or(np.isnan(mins), np.isnan(maxs))[mins != maxs]]
        return np.sum(good) / len(good)

    def compute_metrics(self, predictions: np.ndarray, log: bool = True) -> QMAPMetrics:
        """
        Compute the QMAP metrics for the predictions.
        :param predictions: The predictions to evaluate. It should have the same length and order as this dataset.
        :param log: If true, apply a log10 on the targets.
        :return: The metrics for the predictions.
        """
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        if predictions.ndim > 2:
            raise ValueError("Predictions must have a shape: (N_samples, N_species) if specie_as_input is False or (N_samples,)  otherwise")
        targets = np.log10(self.targets) if log else self.targets
        mse = 0
        mae = 0
        rmse = 0
        r2 = 0
        spearman = 0
        kendalls_tau = 0
        pearson = 0
        for col in range(predictions.shape[1]):
            mse += np.mean((targets[:, col] - predictions[:, col]) ** 2)
            mae += np.mean(np.abs(targets[:, col] - predictions[:, col]))
            rmse += np.sqrt(mse)
            r2 += r2_score(targets[:, col], predictions[:, col])
            spearman += spearmanr(targets[:, col], predictions[:, col]).statistic
            kendalls_tau += kendalltau(targets[:, col], predictions[:, col]).statistic
            pearson += pearsonr(targets[:, col], predictions[:, col]).statistic

        n = predictions.shape[1]
        return QMAPMetrics(split=self.split, threshold=int(100 * self.threshold),
                           rmse=rmse / n,
                           mse=mse / n,
                           mae=mae / n,
                           r2=r2 / n,
                           spearman=spearman / n,
                           kendalls_tau=kendalls_tau / n,
                           pearson=pearson / n)


