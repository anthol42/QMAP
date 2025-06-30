import torch
from typing import Literal

def make_optimizer(parameters, loss_parameters, optimizer: Literal["Adam", "AdamW"],
                   lr: float, weight_decay: float):
    if optimizer not in ["Adam", "AdamW"]:
        raise ValueError(f"Optimizer {optimizer} is not supported. Use 'Adam' or 'AdamW'.")

    params = [
        dict(
            params=parameters,
            weight_decay=weight_decay,
        ),
        dict(
            params=loss_parameters,
            weight_decay=0
        )
    ]
    if optimizer == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr,
            weight_decay=weight_decay)
    else:
        return torch.optim.Adam(
            params, lr=lr,
            weight_decay=weight_decay)

