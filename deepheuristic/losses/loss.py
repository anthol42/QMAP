import torch
from torch import nn
from typing import Optional, Literal

class Criterion(nn.Module):
    def __init__(self, loss_type: Literal['MSE', 'BCE'] = 'MSE'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "MSE":
            self.criterion = nn.MSELoss()
            self.activation = nn.PReLU(init=0.25)
        elif loss_type == "BCE": # TODO: Find why it does not work
            self.criterion = nn.BCELoss()
            self.activation = nn.PReLU(init=-0.5)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


    def forward(self, pred, target):
        # Normalize between 0 and 1 (Soft with PReLU) Because it is included in weight decay, it will converge to a relu
        pred = self.activation(pred)
        return self.criterion(pred, target)
