import optuna
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
    for i, prg, (s1_str, s2_str, seq1, seq2, label) in progress(dataloader, type="dl").enum().ref():
        if epoch == 0 and i == 0 and sample_inputs is not None:
            print()
            log(f"Saving sample inputs at {sample_inputs}")
            torch.save((s1_str, s2_str, seq1, seq2, label), sample_inputs)
        # Setup - Copying to gpu if available
        seq1, seq2, label = seq1.to(device), seq2.to(device), label.to(device)
        # for i in range(10_000):
        optimizer.zero_grad()
        # Training with possibility of mixed precision
        if scaler:
            with torch.autocast(device_type=str(device), dtype=torch.float16):
                pred1 = model(seq1).unsqueeze(1) # Shape(B, 1, E)
                pred2 = model(seq2).unsqueeze(2) # Shape(B, E, 1)
                # Dot product of the vectors to get the predicted labels
                pred = (pred1 @ pred2).squeeze(-1) # Shape(B, 1, 1) => Shape(B, 1)
                loss = criterion(pred, label) # MSE loss between predicted and true labels
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred1 = model(seq1).unsqueeze(1)  # Shape(B, 1, E)
            pred2 = model(seq2).unsqueeze(2)  # Shape(B, E, 1)
            # Dot product of the vectors to get the predicted labels
            pred = (pred1 @ pred2).squeeze(-1) # Shape(B, 1, 1) => Shape(B, 1)
            loss = criterion(pred, label)  # MSE loss between predicted and true labels
            loss.backward()
            optimizer.step()

        State.global_step += 1

        if scheduler:
            scheduler.step()

        # Calculate metrics
        metr = {}
        targets = label.detach().cpu()
        pred = pred.detach().cpu()
        for metric_name, metric_fn in metrics.items():
            metr[metric_name] = metric_fn(targets, pred)

            # print(f"{i} - Loss: {loss.item()}; {','.join(f'{metric}: {fn(targets, pred)}' for metric, fn in metrics.items())}")
        metr["loss"] = loss.item()
        lossCounter(loss)

        # Report metrics
        if i % 100 == 0 and i > 0: # We ignore the first step
            State.resultSocket.add_scalar('Step/loss', metr["loss"], epoch=epoch, step=State.global_step)
            State.resultSocket.add_scalar('Step/alpha', criterion.activation.weight.item(), epoch=epoch, step=State.global_step)
            for metric_name, value in metr.items():
                if metric_name == "loss":
                    continue
                State.resultSocket.add_scalar(f'Step/{metric_name}', value, epoch=epoch, step=State.global_step)

        #Display metrics
        prg.report(
            loss=lossCounter.compute().item(),
            **{k: v.compute().item() for k, v in metrics.items()}
        )
    print()
    # Report epochs metrics
    for metric_name, counter in metrics.items():
        State.resultSocket.add_scalar(f'Train/{metric_name}', counter.compute(), State.global_step, epoch=epoch)
    State.resultSocket.add_scalar(f'Train/loss', lossCounter.compute(), State.global_step, epoch=epoch, flush=True)

@torch.inference_mode()
def validation_step(model, dataloader, criterion, epoch, device, metrics: dict = None, verbose: bool = True):
    if metrics is None:
        metrics = {}
    model.eval()
    lossCounter = DynamicMetric().to(device)
    # Reset metrics
    for m in metrics.values():
        m.reset()

    for _, _, seq1, seq2, label in progress(dataloader, desc="Validating", type="dl"):
        # Setup - Copying to gpu if available
        seq1, seq2, label = seq1.to(device), seq2.to(device), label.to(device)

        # Evaluating
        pred1 = model(seq1).unsqueeze(1)  # Shape(B, 1, E)
        pred2 = model(seq2).unsqueeze(2)  # Shape(B, E, 1)
        # Dot product of the vectors to get the predicted labels
        pred = (pred1 @ pred2).squeeze(-1)  # Shape(B, 1, 1) => Shape(B, 1)
        loss = criterion(pred, label)  # MSE loss between predicted and true labels

        # Calculate metrics
        targets = label.detach().cpu()
        pred = pred.detach().cpu()
        if metrics is not None:
            for metric_name, metric_fn in metrics.items():
                metric_fn(pred, targets)

        # Report metrics
        lossCounter(loss)

    # Display metrics
    if verbose:
        print(f'\r\033[K\r  ✅  {Colors.text}{format_metrics(val=True, loss=lossCounter, **metrics)}{Colors.reset}')

    # Report epochs metrics
    last_valid = {}
    for metric_name, counter in metrics.items():
        State.resultSocket.add_scalar(f'Valid/{metric_name}', counter.compute(), State.global_step, epoch=epoch)
        last_valid[metric_name] = counter.compute()
    State.resultSocket.add_scalar(f'Valid/loss', lossCounter.compute(), State.global_step, epoch=epoch, flush=True)
    last_valid["loss"] = lossCounter.compute()
    State.last_valid = last_valid

def train(model, optimizer, train_loader, val_loader, criterion, num_epochs, device, config, scheduler=None,
          metrics: dict = None, noscaler: bool = False, watch: str = "accuracy", sample_inputs: Optional[str] = None,
          optuna_trial: Optional[optuna.Trial] = None,
          verbose: int = 3):
    log("Training in optuna mode") if optuna_trial is not None else None
    State.global_step = 0
    # Checkpoints
    m, b = ("MIN", float("inf")) if watch == "loss" or "MAE" else ("MAX", float('-inf'))
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

        if optuna_trial is not None:
            # Report the best value to optuna
            optuna_trial.report(State.last_valid[watch], epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

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

    all_preds = []
    for prg, (_, _, seq1, seq2, label) in progress(dataloader, type="dl", desc="Evaluating", end="\n").ref():
        # Setup - Copying to gpu if available
        seq1, seq2, label = seq1.to(device), seq2.to(device), label.to(device)

        # Evaluating
        pred1 = model(seq1).unsqueeze(1)  # Shape(B, 1, E)
        pred2 = model(seq2).unsqueeze(2)  # Shape(B, E, 1)
        # Dot product of the vectors to get the predicted labels
        pred = (pred1 @ pred2).squeeze(-1)  # Shape(B, 1, 1) => Shape(B, 1)
        all_preds.append(pred.detach().cpu())
        loss = criterion(pred, label)  # MSE loss between predicted and true labels

        # Calculate metrics
        targets = label.detach().cpu()
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

    all_preds = torch.cat(all_preds)
    all_preds = criterion.activation(all_preds.to(device)).cpu()
    # Report epochs metrics
    return dict(loss=lossCounter.compute(), **{k: v.compute() for k, v in metrics.items()}), all_preds
