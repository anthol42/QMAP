import torch
from typing import Literal

def make_optimizer(parameters, optimizer: Literal["Adam", "AdamW"],
                   lr: float, weight_decay: float):
    if optimizer not in ["Adam", "AdamW"]:
        raise ValueError(f"Optimizer {optimizer} is not supported. Use 'Adam' or 'AdamW'.")
    if optimizer == "AdamW":
        return torch.optim.AdamW(
            parameters, lr=lr,
            weight_decay=weight_decay)
    else:
        return torch.optim.Adam(
            parameters, lr=lr,
            weight_decay=weight_decay)

