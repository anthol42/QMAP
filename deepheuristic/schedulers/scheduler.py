from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pyutils import ConfigFile
def make_scheduler(optimizer: Optimizer, config: ConfigFile):

    return CosineAnnealingWarmRestarts(optimizer, config["scheduler"]["n_iter_restart"],
                                       config["scheduler"]["factor_increase"], eta_min=config["scheduler"]["min_lr"])

