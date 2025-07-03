import torch
import torch.nn.functional as F
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


    def forward(self, pred1, pred2, target):
        pred = (pred1.unsqueeze(1) @ pred2.unsqueeze(2)).squeeze(-1)
        # Normalize between 0 and 1 (Soft with PReLU) Because it is included in weight decay, it will converge to a relu
        pred = self.activation(pred)
        return self.criterion(pred, target), pred

class LateProjCriterion(nn.Module):
    def __init__(self, activation: nn.Module, loss_type: Literal['MSE', 'BCE'] = 'MSE'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_type == "BCE":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.activation = activation


    def forward(self, pred1, pred2, target):
        score = self.activation(pred1, pred2)
        if self.loss_type == "BCE":
            pred = torch.sigmoid(score)
        else: # MSE
            score = torch.sigmoid(score)
            pred = score
        return self.criterion(score, target), pred