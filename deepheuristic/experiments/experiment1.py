import torch
import os
from data.dataloader import make_dataloader
from models import ESMEncoder
from optimizers.optimizer import make_optimizer
from training.train import train, evaluate
from losses import Criterion
from schedulers.scheduler import make_scheduler
import sys
import shutil
from utils import State, get_profile
from utils.esm_alphabet import ESMAlphabet
from deepboard.resultTable import ResultTable
import utils
from pyutils import Colors, ConfigFile
from utils.bin import *
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef
from torchinfo import summary
# To verify if the config has the good format
from configs.formats import config_format

metrics = {
    "mae": MeanAbsoluteError(),
    "pcc": PearsonCorrCoef(),
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
    config = ConfigFile(args.config, config_format.get(option="ESM"), verify_path=True, profiles=["local", "hpc"])

    config.change_profile(get_profile())
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
        lr=config["training"]["lr"],
        min_lr=config["training"]["min_lr"],
        wd=config["training"]["weight_decay"],
        loss=config["training"]["loss"],
        optimizer=config["training"]["optimizer"],
        head_dropout=config["model"]["head_dropout"],
        proj_dim=config["model"]["proj_dim"],
        pretrained=not args.randominit
    )
    run_id = resultSocket.run_id

    config["model"]["model_dir"] = f'{config["model"]["model_dir"]}/{run_id}'

    State.resultSocket = resultSocket

    alphabet = ESMAlphabet()
    # Loading the data
    train_loader, val_loader, test_loader = make_dataloader(config=config, alphabet=alphabet, fract=args.fract)
    log("Data loaded successfully!")

    # Loading the model
    model = ESMEncoder(
        alphabet=alphabet,
        num_layers = config["model"]["num_layers"],
        embed_dim = config["model"]["embed_dim"],
        attention_heads = config["model"]["attention_heads"],
        token_dropout = config["model"]["token_dropout"],
        attention_dropout = config["model"]["attention_dropout"],
        layer_dropout = config["model"]["layer_dropout"],
        head_dropout = config["model"]["head_dropout"],
        head_dim = config["model"]["embed_dim"],
        head_depth = config["model"]["head_depth"],
        proj_dim = config["model"]["proj_dim"],
        use_clf_token = config["model"]["use_clf_token"]
    )
    if not args.randominit:
        log("Loading pretrained weights")
        size = config["model"]["name"].split("_")[-1]
        utils.load_weights(model.backbone, size, config["model"]["weights_path"])
    model.to(device)
    if args.verbose >= 3:
        B = config["data"]["batch_size"]
        L = 100 # Max length of the sequence
        summary(model, input_data=(torch.randint(0, 20, (B, L))), device=device)
    log("Model loaded successfully!")

    # Loading optimizer, loss and scheduler
    loss = Criterion(loss_type=config["training"]["loss"])
    loss.to(device)
    optimizer = make_optimizer(model.parameters(),
                               loss.parameters(),
                               config["training"]["optimizer"],
                               lr=config["training"]["lr"],
                               weight_decay=config["training"]["weight_decay"])

    scheduler = make_scheduler(optimizer, config, num_steps=config["training"]["num_epochs"] * len(train_loader))

    # Training
    # Prepare the path of input sampling if flag is set
    if args.sample_inputs:
        sample_inputs = f"{config['model']['model_dir']}/inputs.pth"
    else:
        sample_inputs = None
    log("Begining training...")
    log(f"Watching: {args.watch}")
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
        "activation": loss.activation.state_dict(),
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
    resultSocket.write_result(**{name: result.item() for name, result in results.items()})




