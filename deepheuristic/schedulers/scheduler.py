from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from pyutils import ConfigFile
def make_scheduler(optimizer: Optimizer, config: ConfigFile, num_steps: int):

    return CosineAnnealingLR(optimizer, eta_min=config["training"]["min_lr"], T_max=num_steps)

