import torch
import utils
from pyutils import progress
from utils import State, DynamicMetric, format_metrics
from typing import *
from utils.bin import *

def train_one_epoch(dataloader, model, optimizer, criterion, epoch, device, scheduler=None, scaler=None,
                    metrics: dict = None, sample_inputs: Optional[str] = None):
    if metrics is None:
        metrics = {}
    model.train()
    lossCounter = DynamicMetric().to(device)
    # Reset metrics
    for m in metrics.values():
        m.reset()
    prg: progress
    for i, prg, (X, y) in progress(dataloader, type="dl").enum().ref():
        if epoch == 0 and i == 0 and sample_inputs is not None:
            print()
            log(f"Saving sample inputs at {sample_inputs}")
            torch.save((X, y), sample_inputs)
        # Setup - Copying to gpu if available
        X, y = X.to(device), y.to(device)
        # for i in range(10_000):
        optimizer.zero_grad()
        # Training with possibility of mixed precision
        if scaler:
            with torch.autocast(device_type=str(device), dtype=torch.float16):
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        State.global_step += 1

        if scheduler:
            scheduler.step()

        # Calculate metrics
        metr = {}
        targets = y.detach().cpu()
        pred = pred.detach().cpu()
        for metric_name, metric_fn in metrics.items():
            # print(targets.shape, pred.shape)
            metr[metric_name] = metric_fn(pred, targets)

            # print(f"{i} - Loss: {loss.item()}; {','.join(f'{metric}: {fn(targets, pred)}' for metric, fn in metrics.items())}")
        metr["loss"] = loss.item()

        # Report metrics
        State.resultSocket.add_scalar('Step/Loss', metr["loss"], epoch=epoch)
        lossCounter(loss)
        for metric_name, value in metr.items():
            if metric_name == "loss":
                continue
            State.resultSocket.add_scalar(f'Step/{metric_name}', value, epoch=epoch)

        #Display metrics
        prg.report(
            loss=lossCounter.compute().item(),
            **{k: v.compute().item() for k, v in metrics.items()}
        )

    # Report epochs metrics
    for metric_name, counter in metrics.items():
        State.resultSocket.add_scalar(f'Train/{metric_name}', counter.compute(), State.global_step, epoch=epoch)
    State.resultSocket.add_scalar(f'Train/Loss', lossCounter.compute(), State.global_step, epoch=epoch, flush=True)

@torch.inference_mode()
def validation_step(model, dataloader, criterion, epoch, device, metrics: dict = None, verbose: bool = True):
    if metrics is None:
        metrics = {}
    model.eval()
    lossCounter = DynamicMetric().to(device)
    # Reset metrics
    for m in metrics.values():
        m.reset()

    for X, y in dataloader:
        # Setup - Copying to gpu if available
        X, y = X.to(device), y.to(device)

        # Evaluating
        pred = model(X)
        loss = criterion(pred, y)

        # Calculate metrics
        targets = y.detach().cpu()
        pred = pred.detach().cpu()
        if metrics is not None:
            for metric_name, metric_fn in metrics.items():
                metric_fn(pred, targets)

        # Report metrics
        lossCounter(loss)

    # Display metrics
    if verbose:
        print(f'{Colors.darken} | {format_metrics(val=True, loss=lossCounter, **metrics)}{Colors.reset}')

    # Report epochs metrics
    last_valid = {}
    for metric_name, counter in metrics.items():
        State.resultSocket.add_scalar(f'Valid/{metric_name}', counter.compute(), State.global_step, epoch=epoch)
        last_valid[metric_name] = counter.compute()
    State.resultSocket.add_scalar(f'Valid/Loss', lossCounter.compute(), State.global_step, epoch=epoch, flush=True)
    last_valid["loss"] = lossCounter.compute()
    State.last_valid = last_valid

def train(model, optimizer, train_loader, val_loader, criterion, num_epochs, device, config, scheduler=None,
          metrics: dict = None, noscaler: bool = False, watch: str = "accuracy", sample_inputs: Optional[str] = None,
          verbose: int = 3):
    State.global_step = 0
    # Checkpoints
    m, b = ("MIN", float("inf")) if watch == "loss" else ("MAX", float('-inf'))
    save_best_model = utils.SaveBestModel(
        config["model"]["model_dir"], metric_name=f"validation {watch}", model_name=config["model"]["name"],
        best_metric_val=b, evaluation_method=m, verbose=verbose == 3)
    if str(device) == "cuda" and not noscaler:
        # For mixed precision training
        scaler = torch.amp.GradScaler()
    else:
        # We cannot do mixed precision on cpu or mps
        scaler = None

    for epoch in range(1, num_epochs + 1):
        # Setup
        print(f"Epoch {epoch}/{num_epochs}") if verbose == 3 else None

        # Train the epoch and validate
        train_one_epoch(
            train_loader, model, optimizer, criterion, epoch, device, scheduler, scaler, metrics, sample_inputs=sample_inputs
        )
        validation_step(
            model, val_loader, criterion, epoch, device, metrics, verbose == 3
        )

        # Checkpoint
        save_best_model(State.last_valid[watch], epoch, model, optimizer, criterion)

@torch.inference_mode()
def evaluate(model, dataloader, criterion, device, metrics: dict = None):
    if metrics is None:
        metrics = {}
    model.eval()
    lossCounter = DynamicMetric().to(device)
    # Reset metrics
    for m in metrics.values():
        m.reset()

    output_rate = 25 if str(device) == 'cuda' else 1
    for prg, (X, y) in progress(dataloader, type="dl", desc="Evaluating", end="\n").ref():
        # Setup - Copying to gpu if available
        X, y = X.to(device), y.to(device)

        # Evaluating
        pred = model(X)
        loss = criterion(pred, y)

        # Calculate metrics
        targets = y.detach().cpu()
        pred = pred.detach().cpu()
        for metric_name, metric_fn in metrics.items():
            metric_fn(pred, targets)

        # Report metrics
        lossCounter(loss)
        # Display metrics
        prg.report(
            loss=lossCounter.compute().item(),
            **{k: v.compute().item() for k, v in metrics.items()}
        )

    # Report epochs metrics
    return dict(loss=lossCounter.compute(), **{k: v.compute() for k, v in metrics.items()})
