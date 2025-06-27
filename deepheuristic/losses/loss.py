import torch

def make_loss(config):
    return torch.nn.CrossEntropyLoss()

