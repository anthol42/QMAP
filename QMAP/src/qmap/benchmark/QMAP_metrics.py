from dataclasses import dataclass
import numpy as np
from typing import Literal

@dataclass
class QMAPRegressionMetrics:
    split: int
    threshold: Literal[55, 60]
    rmse: float
    mse: float
    mae: float
    r2: float
    spearman: float
    kendalls_tau: float
    pearson: float

    @property
    def md_row(self):
        """
        Return a row of a markdown table containing the metrics values rounded to 4 decimal points.
        """
        return f'| {self.split} | {self.threshold} | {self.rmse:.4f} | {self.mse:.4f} | {self.mae:.4f} | {self.r2:.4f} | {self.spearman:.4f} | {self.kendalls_tau:.4f} | {self.pearson:.4f} |\n'

    @property
    def md_col(self):
        """
        Return a column of a markdown table containing the metrics values
        """
        return f'| Split | Threshold | RMSE | MSE | MAE | R2 | Spearman | Kendall\'s Tau | Pearson |\n' \
               f'|-------|-----------|------|-----|-----|----|----------|----------------|---------|\n'

    def dict(self):
        """
        Return a dictionary containing the metrics values.
        """
        return dict(split=self.split,threshold=self.threshold,rmse=self.rmse,mse=self.mse,mae=self.mae,
                    r2=self.r2,spearman=self.spearman,kendalls_tau=self.kendalls_tau,pearson=self.pearson)
    def __repr__(self):
        return f'QMAPMetrics(split: {self.split}, threshold: {self.threshold}%)'

    def __str__(self):
        return f'QMAPMetrics(split: {self.split}, threshold: {self.threshold}%):\n - RMSE: {self.rmse:.4f}\n - MSE: {self.mse:.4f}\n - MAE: {self.mae:.4f}\n - R2: {self.r2:.4f}\n - Spearman: {self.spearman:.4f}\n - Kendall\'s Tau: {self.kendalls_tau:.4f}\n - Pearson: {self.pearson:.4f}'


@dataclass
class QMAPClassificationMetrics:
    split: int
    threshold: Literal[55, 60]
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float

    @property
    def md_row(self):
        """
        Return a row of a markdown table containing the metrics values rounded to 4 decimal points.
        """
        return f'| {self.split} | {self.threshold} | {self.balanced_accuracy:.4f} | {self.precision:.4f} | {self.recall:.4f} | {self.f1:.4f} | {self.mcc:.4f} |\n'

    @property
    def md_col(self):
        """
        Return a column of a markdown table containing the metrics values
        """
        return f'| Split | Threshold | Balanced Accuracy | Precision | Recall | F1 | MCC |\n' \
               f'|-------|-----------|-------------------|-----------|--------|----|-----|\n'

    def dict(self):
        """
        Return a dictionary containing the metrics values.
        """
        return dict(split=self.split,threshold=self.threshold,balanced_accuracy=self.balanced_accuracy,precision=self.precision,recall=self.recall,
                    f1=self.f1, mcc=self.mcc)

    def __repr__(self):
        return f'QMAPMetrics(split: {self.split}, threshold: {self.threshold}%)'

    def __str__(self):
        return f'QMAPMetrics(split: {self.split}, threshold: {self.threshold}%):\n - Balanced Accuracy: {self.balanced_accuracy:.4f}\n - Precision: {self.precision:.4f}\n - Recall: {self.recall:.4f}\n - F1: {self.f1:.4f}\n - MCC: {self.mcc:.4f}\n'
