import torch
import utils
from pyutils import progress
from utils import DynamicMetric, format_metrics
from utils.bin import *

def train_one_epoch(dataloader, model, ema_model, optimizer, criterion, epoch, device, resultSocket, start_step: int, scheduler=None, scaler=None,
                    metrics: dict = None):

    if metrics is None:
        metrics = {}
    model.train()
    lossCounter = DynamicMetric().to(device)
    # Reset metrics
    for m in metrics.values():
        m.reset()
    prg: progress
    for i, prg, (s1_str, s2_str, seq1, seq2, label) in progress(dataloader, type="dl").enum().ref():
        # Setup - Copying to gpu if available
        seq1, seq2, label = seq1.to(device), seq2.to(device), label.to(device)
        optimizer.zero_grad()
        # Training with possibility of mixed precision
        if scaler:
            with torch.autocast(device_type=str(device), dtype=torch.float16):
                pred1 = model(seq1)
                pred2 = model(seq2)
                loss, pred = criterion(pred1, pred2, label) # MSE loss between predicted and true labels=
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred1 = model(seq1)
            pred2 = model(seq2)
            loss, pred = criterion(pred1, pred2, label)  # MSE loss between predicted and true labels
            loss.backward()
            optimizer.step()

        ema_model and ema_model.update()


        if scheduler:
            scheduler.step()

        # Calculate metrics
        metr = {}
        targets = label.detach().cpu()
        pred = pred.detach().cpu()
        for metric_name, metric_fn in metrics.items():
            metr[metric_name] = metric_fn(targets, pred)
        metr["loss"] = loss.item()
        lossCounter(loss)

        # Report metrics
        if i % 100 == 0 and i > 0: # We ignore the first step
            resultSocket.add_scalar('Step/loss', metr["loss"], epoch=epoch, step=start_step + i)
            for metric_name, value in metr.items():
                if metric_name == "loss":
                    continue
                resultSocket.add_scalar(f'Step/{metric_name}', value, epoch=epoch, step=start_step + i)

        #Display metrics
        prg.report(
            loss=lossCounter.compute().item(),
            **{k: v.compute().item() for k, v in metrics.items()}
        )

    print()
    # Report epochs metrics
    for metric_name, counter in metrics.items():
        resultSocket.add_scalar(f'Train/{metric_name}', counter.compute(), start_step + i, epoch=epoch)
    resultSocket.add_scalar(f'Train/loss', lossCounter.compute(), start_step + i, epoch=epoch, flush=True)
    return i # Number of steps run

@torch.inference_mode()
def validation_step(model, dataloader, criterion, epoch, device, resultSocket, global_step: int, metrics: dict = None, verbose: bool = True):
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
        pred1 = model(seq1)
        pred2 = model(seq2)
        loss, pred = criterion(pred1, pred2, label)  # MSE loss between predicted and true labels

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
        resultSocket.add_scalar(f'Valid/{metric_name}', counter.compute(), global_step, epoch=epoch)
        last_valid[metric_name] = counter.compute()
    resultSocket.add_scalar(f'Valid/loss', lossCounter.compute(), global_step, epoch=epoch, flush=True)
    last_valid["loss"] = lossCounter.compute()
    return last_valid

def train(model, ema_model, optimizer, train_loader, val_loader, criterion, num_epochs, device, config, resultSocket, scheduler=None,
          metrics: dict = None, verbose: int = 3):
    global_step = 0
    # Checkpoints
    m, b = ("MIN", float("inf"))
    save_best_model = utils.SaveBestModel(
        config["model"]["model_dir"], metric_name=f"validation mae", model_name=config["model"]["name"],
        best_metric_val=b, evaluation_method=m, verbose=verbose == 3)
    if str(device) == "cuda":
        # For mixed precision training
        scaler = torch.amp.GradScaler()
    else:
        # We cannot do mixed precision on cpu or mps
        scaler = None

    for epoch in range(1, num_epochs + 1):
        # Setup
        print(f"Epoch {epoch}/{num_epochs}") if verbose == 3 else None

        # Train the epoch and validate
        global_step += train_one_epoch(train_loader, model, ema_model, optimizer, criterion, epoch, device, resultSocket, global_step, scheduler, scaler, metrics)
        last_valid = validation_step(
            ema_model, val_loader, criterion, epoch, device, resultSocket, global_step, metrics, verbose == 3
        )

        # Checkpoint
        save_best_model(last_valid['mae'], epoch, model, ema_model, optimizer, criterion)

    return global_step

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
        pred1 = model(seq1)
        pred2 = model(seq2)
        loss, pred = criterion(pred1, pred2, label)  # MSE loss between predicted and true labels

        # Dot product of the vectors to get the predicted labels
        all_preds.append(pred.detach().cpu())

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
    # Report epochs metrics
    return dict(loss=lossCounter.compute(), **{k: v.compute() for k, v in metrics.items()}), all_preds
