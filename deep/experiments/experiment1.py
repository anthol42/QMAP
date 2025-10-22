import torch
from data.dataloader import make_dataloader
from models import ESMEncoder
from training.train import train, evaluate
from loss import Criterion
from utils import experiment
from deepboard.resultTable import ResultTable, NoCommitAction
import utils
from pyutils import ConfigFile
from ema_pytorch import EMA
from utils.bin import *
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef
from configs.formats import config_format
import matplotlib.pyplot as plt
import numpy as np
import typer

plt.rcParams['savefig.dpi'] = 600

metrics = {
    "mae": MeanAbsoluteError(),
    "pcc": PearsonCorrCoef(),
}

@experiment
def experiment1(ctx: typer.Context,
                config_path: str = typer.Argument(..., help="Path to the configuration file"),
                cpu: bool = typer.Option(False, help="Force the use of cpu even if a gpu is available"),
                fract: float = typer.Option(1.0, help="Fraction of the dataset to use for training"),
                debug: bool = typer.Option(False, help="Debug mode, meaning that logs are not permanently saved"),
                verbose: int = typer.Option(3, help="Verbose level from 0 to 3"),
                comment: str = typer.Option(None, help="Comment to add to the result log")):
    kwargs = utils.parse_ctx(ctx.args)
    config_loggers_with_verbose(verbose)
    DEBUG = debug

    # Setup
    device = utils.get_device(cpu)
    log(f"Running on {device}")

    # Loading the config file
    config = ConfigFile(config_path, config_format, verify_path=True)

    # Enable to override config from command line by passing parameters in the following format: --config.key1.key2=value
    config.override_config(kwargs)

    # Preparing Result Table
    # The result table will store the logs of our training. (Like tensorboard, but better:))
    # We can view those logs with cli: ```deepboard results/resultTable.db```
    rtable = ResultTable("results/resultTable.db", nocommit_action=NoCommitAction.RAISE)
    if DEBUG:
        log(f"Running in {Colors.warning}DEBUG{Colors.reset} mode!")
        resultSocket = rtable.new_debug_run(utils.get_experiment_name(__name__), config_path, {}, comment=comment)
    else:
        resultSocket = rtable.new_run(utils.get_experiment_name(__name__), config_path, {}, comment=comment)


    # Avoid overriding previous runs, we create a new folder with the run id
    config["model"]["model_dir"] = f'{config["model"]["model_dir"]}/{resultSocket.run_id}'

    # Loading the data
    train_loader, val_loader, test_loader = make_dataloader(config=config, fract=fract)
    log("Data loaded successfully!")

    # Loading the model
    model = ESMEncoder()
    ema_model = EMA(
        model,
        beta = 0.9999,              # exponential moving average factor
        update_after_step = 500,    # only after this number of .update() calls will it start updating
        update_every = 10,
    )
    log("Loading pretrained weights")
    utils.load_weights(model.backbone, "35M", config["model"]["weights_path"])
    model.to(device)
    ema_model.to(device)
    log("Model loaded successfully!")

    # Loading optimizer, loss and scheduler
    loss = Criterion(
        smoothness=config["training"]["smoothness"],
        diversity=config["training"]["diversity"],
        var=config["training"]["var"],
        orthogonality=config["training"]["orthogonality"],
    )
    loss.to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss.parameters()),  lr=config["training"]["lr"], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           eta_min=0.,
                                                           T_max=config["training"]["num_epochs"] * len(train_loader)
                                                           )

    # Training
    log("Begining training...")
    global_step = train(
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss,
        num_epochs=config["training"]["num_epochs"],
        device=device,
        scheduler=scheduler,
        config=config,
        resultSocket=resultSocket,
        metrics=metrics,
        verbose=verbose,
    )
    log("Training done!")

    # Load best model
    log("Loading best model")
    checkpoint = torch.load(f'{config["model"]["model_dir"]}/{config["model"]["name"]}.pth', weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    log("Loading best ema model")
    ema_model.load_state_dict(checkpoint["ema_state_dict"])

    # Test
    results, all_preds = evaluate(ema_model,
                                  test_loader, loss, device, metrics=metrics)
    log("Training done!  Saving...")


    # Next, we create figures and fragments to visualize the model performance

    # Error distribution
    error = test_loader.dataset.label - all_preds.numpy().squeeze()
    plt.hist(error, bins=100, density=True)
    plt.title("Error between true identity and predicted identity")
    plt.xlabel("Error (GT - pred)")
    plt.ylabel("Density")
    plt.grid()
    resultSocket.detect_and_log_figures(step=global_step, split="test", epoch=config["training"]["num_epochs"])
    plt.close()

    # Error quantiles table
    abs_error = np.abs(error)
    labels = [0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantiles = [f"{value:.3f}" if abs(value) >= 0.001 else f"{value:.2e}"
             for value in np.quantile(abs_error, labels)]
    rows = [[label, value] for label, value in zip(labels, quantiles)]
    html_table = utils.make_table(["Quantile", "Error"], rows)

    resultSocket.add_fragment(html_table, step=global_step, split="test", epoch=config["training"]["num_epochs"])

    # Save results to the result table
    resultSocket.write_result(**{name: result.item() for name, result in results.items()})