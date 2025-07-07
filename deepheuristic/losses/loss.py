import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Literal

from .functional import diversity_loss, variance_diversity_loss, orthogonality_loss

class Criterion(nn.Module):
    def __init__(self, loss_type: Literal['MSE', 'BCE'] = 'MSE', diversity: float = 0., var: float = 0.,
                 orthogonality: float = 0., smoothness: float = 0.):
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

        self.diversity = diversity
        self.var = var
        self.orthogonality = orthogonality
        self.smoothness = smoothness
        if smoothness > 0:
            self.smooth_activation = nn.PReLU(init=0.25)

    def forward(self, pred1, pred2, target):
        pred = (pred1.unsqueeze(1) @ pred2.unsqueeze(2)).squeeze(-1)
        # Normalize between 0 and 1 (Soft with PReLU) Because it is included in weight decay, it will converge to a relu
        pred = self.activation(pred)
        loss = self.criterion(pred, target)
        embs = torch.cat([pred1, pred2], dim=0)
        if self.smoothness > 0:
            l2 = torch.norm(pred1 - pred2, p=2, dim=1)
            activated = self.smooth_activation(l2)
            added = self.smoothness * ((activated - target)**2).mean()
            loss += added
        if self.diversity > 0:
            added = self.diversity * diversity_loss(embs)
            loss += added
        if self.var > 0:
            added = self.var * 5 * variance_diversity_loss(embs) # Multiply by 5 to be in the same order as the loss
            loss += added
        if self.orthogonality > 0:
            added = self.orthogonality * 100 * orthogonality_loss(embs) # Multiply by 100 to be in the same order as the loss
            loss += added
        return loss, pred

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