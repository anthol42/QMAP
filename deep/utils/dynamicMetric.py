import numpy as np
from torchmetrics import Metric
import torch

class DynamicMetric(Metric):
    """
    Subclass of Metric from torchmetric, compatible with the Feedback API.
    It creates a mean over the arguments passed as input.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("values", default=torch.tensor(0.), dist_reduce_fx="sum") # To mean over all values
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor) -> None:
        values, values = self._input_format(values, values)

        self.values += torch.sum(values)
        self.count += values.numel()
        return torch.tensor(1.)

    def compute(self) -> torch.Tensor:
        return self.values / self.count

    def _input_format(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Implement in the child class to verify the format of the input
        :raise ValueError: If the input is not in the correct format
        """
        return preds, target


