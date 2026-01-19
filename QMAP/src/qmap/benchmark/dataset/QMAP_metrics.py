from dataclasses import dataclass
from typing import Literal

@dataclass
class QMAPRegressionMetrics:
    property_name: str
    n: int
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
        return f'| {self.property_name} | {self.n} | {self.rmse:.4f} | {self.mse:.4f} | {self.mae:.4f} | {self.r2:.4f} | {self.spearman:.4f} | {self.kendalls_tau:.4f} | {self.pearson:.4f} |\n'

    @property
    def md_col(self):
        """
        Return a column of a markdown table containing the metrics values
        """
        return f'| Property | N | RMSE | MSE | MAE | R2 | Spearman | Kendall\'s Tau | Pearson |\n' \
               f'|----------|---|------|-----|-----|----|----------|----------------|---------|\n'

    def dict(self):
        """
        Return a dictionary containing the metrics values.
        """
        return dict(property=self.property_name, n=self.n, rmse=self.rmse,mse=self.mse,mae=self.mae,
                    r2=self.r2,spearman=self.spearman,kendalls_tau=self.kendalls_tau,pearson=self.pearson)
    def __repr__(self):
        return f'QMAPMetrics(property: {self.property_name}; {self.n})'

    def __str__(self):
        return f'QMAPMetrics(property: {self.property_name}; {self.n}):\n - RMSE: {self.rmse:.4f}\n - MSE: {self.mse:.4f}\n - MAE: {self.mae:.4f}\n - R2: {self.r2:.4f}\n - Spearman: {self.spearman:.4f}\n - Kendall\'s Tau: {self.kendalls_tau:.4f}\n - Pearson: {self.pearson:.4f}'

