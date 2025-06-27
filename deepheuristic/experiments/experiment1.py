import torch
import os
from data.dataloader import make_dataloader
from models import Classifier
from optimizers.optimizer import make_optimizer
from training.train import train, evaluate
from losses.loss import make_loss
from schedulers.scheduler import make_scheduler
import sys
import shutil
from utils import State, get_profile
from deepboard.resultTable import ResultTable
import utils
from pyutils import Colors, ConfigFile
from utils.bin import *
from torchmetrics import Accuracy
from torchinfo import summary
# To verify if the config has the good format
from configs.formats import config_format

metrics = {
    "accuracy": Accuracy(task="multiclass", num_classes=10),
}


def experiment1(args, kwargs):
    config_loggers_with_verbose(args.verbose)
    # Setup
    device = utils.get_device(args.cpu)
    log(f"Running on {device}")
    hyper = utils.clean_dict(vars(args).copy())

    DEBUG = args.debug

    # Loading the config file
    # We select the config for the CNN model and the local profile. You can change according to your setup
    config = ConfigFile(args.config, config_format.get(option="CNN"), verify_path=True, profiles=["cpu", "gpu"])

    config.change_profile(get_profile(device))
    config.override_config(kwargs)
    hyper.update(kwargs)

    # Preparing Result Table
    rtable = ResultTable("results/resultTable.db")
    if DEBUG:
        log(f"Running in {Colors.warning}DEBUG{Colors.reset} mode!")
        resultSocket = rtable.new_debug_run(utils.get_experiment_name(__name__), args.config, cli=hyper, comment=args.comment)
    else:
        resultSocket = rtable.new_run(utils.get_experiment_name(__name__), args.config, cli=hyper, comment=args.comment)

    # Add hyperparameters
    resultSocket.add_hparams(
        lr=config["training"]["learning_rate"],
        wd=config["training"]["weight_decay"],
        min_lr=config["scheduler"]["min_lr"],
        dropout2d=config["model"]["dropout2d"],
        dropout=config["model"]["dropout"]
    )
    run_id = resultSocket.run_id

    config["model"]["model_dir"] = f'{config["model"]["model_dir"]}/{run_id}'

    State.resultSocket = resultSocket
    # Loading the data
    train_loader, val_loader, test_loader = make_dataloader(config=config)
    log("Data loaded successfully!")

    # Loading the model
    model = Classifier(config)
    model.to(device)
    if args.verbose >= 3:
        summary(model, input_size=(config["data"]["batch_size"], 1, 28, 28), device=device)
    log("Model loaded successfully!")

    # Loading optimizer, loss and scheduler
    optimizer = make_optimizer(model.parameters(), config)
    loss = make_loss(config)
    scheduler = make_scheduler(optimizer, config)

    # Training
    # Prepare the path of input sampling if flag is set
    if args.sample_inputs:
        sample_inputs = f"{config['model']['model_dir']}/inputs.pth"
    else:
        sample_inputs = None
    log("Begining training...")
    log(f"Watching: {args.watch}")
    try:
        train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=loss,
            num_epochs=config["training"]["num_epochs"],
            device=device,
            scheduler=scheduler,
            config=config,
            metrics=metrics,
            watch=args.watch,
            sample_inputs=sample_inputs,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        log("Keyboard Interrupt detected. Ending training...", start="\n", end="\n\n")

    log("Training done!")

    # Load best model
    log("Loading best model")
    weights = torch.load(f'{config["model"]["model_dir"]}/{config["model"]["name"]}.pth', weights_only=False)[
        "model_state_dict"]
    model.load_state_dict(weights)
    # Test
    results = evaluate(model, test_loader, loss, device, metrics=metrics)
    log("Training done!  Saving...")

    save_dict = {
        "epoch": config["training"]["num_epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_results": results
    }
    torch.save(
        save_dict, f"{config['model']['model_dir']}/final_{config['model']['name']}.pth")
    # Copy config file to model dir
    shutil.copy(args.config, config['model']["model_dir"])

    # Print stats of code
    if config.have_warnings():
        warn(config.get_warnings())

    # Save results
    resultSocket.write_result(accuracy=results["accuracy"].item(), crossEntropy=results["loss"].item())




