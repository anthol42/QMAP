from .dataset import AlignmentDataset
from .alignment_collator import AlignmentCollator
from torch.utils.data import DataLoader
from utils.bin import *

def make_dataloader(config, fract: float):
    train_ds = AlignmentDataset(config, split="train", fract=fract)
    val_ds = AlignmentDataset(config, split="val", fract=fract)
    test_ds = AlignmentDataset(config, split="test")
    # Print dataset sizes
    log(f"Dataset sizes:\n\t-Train: {len(train_ds)}\n\t-Val: {len(val_ds)}\n\t-Test: {len(test_ds)}")
    # Create dataloaders

    # Max length is set to 100 even though we have few sequences that long
    collator = AlignmentCollator()

    train_dl  = DataLoader(train_ds,
                           collate_fn=collator,
                           num_workers=config["data"]["num_workers"],
                           persistent_workers=config["data"]["num_workers"] > 0,
                           batch_size=config["data"]["batch_size"],
                           shuffle=config["data"]["shuffle"],
                           )
    val_dl    = DataLoader(val_ds,
                            collate_fn=collator,
                            num_workers=config["data"]["num_workers"],
                            persistent_workers=config["data"]["num_workers"] > 0,
                            batch_size=config["data"]["batch_size"],
                            shuffle=False)
    test_dl   = DataLoader(test_ds,
                            collate_fn=collator,
                            num_workers=config["data"]["num_workers"],
                            persistent_workers=config["data"]["num_workers"] > 0,
                            batch_size=config["data"]["batch_size"],
                            shuffle=False)
    return train_dl, val_dl, test_dl

