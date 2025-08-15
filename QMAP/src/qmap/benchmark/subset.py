from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Literal, Union
import numpy as np
from .QMAP_metrics import QMAPRegressionMetrics, QMAPClassificationMetrics
from scipy.stats import spearmanr, kendalltau, pearsonr
from .metrics import balanced_accuracy, precision, recall, f1_score, mcc_score, r2_score

class BenchmarkSubset(Dataset):
    """
    Base class of the QMAP benchmark class. It provides a common interface for the benchmark dataset and the subsets.
    """
    def __init__(self, split: int, threshold: Literal[55, 60], dataset_type: Literal['MIC', 'Hemolytic', 'Cytotoxic'],
                     sequences: List[str], species: Optional[List[str]], targets: List[float], c_termini: List[str],
                     n_termini: List[str], unusual_aa: List[dict[int, str]], max_targets: List[float],
                     min_targets: List[float],
                     *,
                     modified_termini: bool = False,
                     allow_unusual_aa: bool = False,
                     specie_as_input: bool = False,
                 ):
        self.split = split
        self.threshold = threshold / 100
        self.dataset_type = dataset_type

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
        """
        Return a numpy array of the targets. Its shape will be (N_samples, N_species) if specie_as_input is False or (N_samples,) otherwise.
        """
        return np.array(self._targets)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
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
        provided. This method only work with MIC datasets (It does not work with Hemolytic or Cytotoxic datasets).
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

    def compute_metrics(self, predictions: np.ndarray, log: bool = True) -> Union[QMAPRegressionMetrics, QMAPClassificationMetrics]:
        """
        Compute the QMAP metrics given the predictions of the model. If the dataset type is MIC, it will return the
        following metrics:
        - RMSE
        - MSE
        - MAE
        - R2
        - Spearman correlation
        - Kendall's tau
        - Pearson correlation

        If the dataset type is Hemolytic or Cytotoxic, it will return the following metrics:
        - Balanced accuracy
        - Precision
        - Recall
        - F1 score
        - Matthews correlation coefficient [MCC]

        Note:

            This does not include the accuracy metric, which is computed separately.

        :param predictions: The predictions to evaluate. It should have the same length and order as this dataset.
        :param log: If true, apply a log10 on the targets.
        :return: A QMAPMetrics object containing all the metrics.
        """
        if self.dataset_type == "MIC":
            return self._regression_metrics(predictions, log=log)
        else:
            return self._classification_metrics(predictions)

    def _classification_metrics(self, predictions: np.ndarray) -> QMAPClassificationMetrics:
        """
        Compute the QMAP metrics given the predictions of the model. The metrics computed are:
        - Balanced accuracy
        - Precision
        - Recall
        - F1 score
        - Matthews correlation coefficient [MCC]

        :param predictions: The predictions can be a probability array (0-1) or a binary array {0, 1}. It should have the same length and order as this dataset. If float values are provided, they will be thresholded at 0.5.
        :return: A QMAPClassificationMetrics object containing all the metrics.
        """
        if len(predictions.shape) > 2:
            raise ValueError("Predictions should be an array of shape (N_samples, ) or (N_samples, 1)")

        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            raise ValueError("If 2D, the predictions should be of shape (N_samples, 1) for binary classification.")
        else:
            predictions = predictions.reshape(-1)

        if predictions.dtype == np.float16 or predictions.dtype == np.float32 or predictions.dtype == np.float64:
            predictions = (predictions > 0.5).astype(int)

        targets = self.targets.reshape(-1)

        return QMAPClassificationMetrics(split=self.split, threshold=int(100 * self.threshold),
                           balanced_accuracy=balanced_accuracy(targets, predictions),
                           precision=precision(targets, predictions),
                           recall=recall(targets, predictions),
                           f1=f1_score(targets, predictions),
                           mcc=mcc_score(targets, predictions))

    def _regression_metrics(self, predictions: np.ndarray, log: bool = True) -> QMAPRegressionMetrics:
        """
        Compute the QMAP metrics given the predictions of the model. The metrics computed are:
        - RMSE
        - MSE
        - MAE
        - R2
        - Spearman correlation
        - Kendall's tau
        - Pearson correlation

        Note:

            This does not include the accuracy metric, which is computed separately.

        :param predictions: The predictions to evaluate. It should have the same length and order as this dataset.
        :param log: If true, apply a log10 on the targets.
        :return: A QMAPRegressionMetrics object containing all the metrics.
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
        return QMAPRegressionMetrics(split=self.split, threshold=int(100 * self.threshold),
                           rmse=rmse / n,
                           mse=mse / n,
                           mae=mae / n,
                           r2=r2 / n,
                           spearman=spearman / n,
                           kendalls_tau=kendalls_tau / n,
                           pearson=pearson / n)

