import torch

def make_optimizer(parameters, config):
    return torch.optim.Adam(
        parameters, lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"])

