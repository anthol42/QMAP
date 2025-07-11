import torch
import os
from data.dataloader import make_dataloader, make_rnd_test_loader
from models import ESMEncoder
from optimizers.optimizer import make_optimizer
from training.train import train, evaluate
from losses import Criterion, LateProjCriterion
from schedulers.scheduler import make_scheduler
from typing import Optional
import shutil
from utils import State, get_profile
from utils.esm_alphabet import ESMAlphabet
from deepboard.resultTable import ResultTable, NoCommitAction
import utils
from pyutils import Colors, ConfigFile
from ema_pytorch import EMA
from utils.bin import *
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef
from torchinfo import summary
# To verify if the config has the good format
from configs.formats import config_format
import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml

plt.rcParams['savefig.dpi'] = 600

metrics = {
    "mae": MeanAbsoluteError(),
    "pcc": PearsonCorrCoef(),
}


def experiment1(args, kwargs, config: Optional[ConfigFile] = None, trial: Optional[optuna.Trial] = None):
    config_loggers_with_verbose(args.verbose)
    # Setup
    device = utils.get_device(args.cpu)
    log(f"Running on {device}")
    hyper = utils.clean_dict(vars(args).copy())

    # Loading the config file
    # We select the config for the CNN model and the local profile. You can change according to your setup
    if config is None:
        OPTUNA = False
        config = ConfigFile(args.config, config_format.get(option="ESM"), verify_path=True, profiles=["local", "hpc"])

        config.change_profile(get_profile())
        config.override_config(kwargs)
    else:
        OPTUNA = True
        log("Training with optuna, using config override")

    DEBUG = args.debug or OPTUNA
    hyper.update(kwargs)

    # Preparing Result Table
    rtable = ResultTable("results/resultTable.db", nocommit_action=NoCommitAction.RAISE)
    if DEBUG:
        log(f"Running in {Colors.warning}DEBUG{Colors.reset} mode!")
        resultSocket = rtable.new_debug_run(utils.get_experiment_name(__name__), args.config, cli=hyper, comment=args.comment, disable=OPTUNA)
    else:
        resultSocket = rtable.new_run(utils.get_experiment_name(__name__), args.config, cli=hyper, comment=args.comment, disable=OPTUNA)

    # Add hyperparameters
    resultSocket.add_hparams(
        dataset = config["data"]["dataset"],
        lr=config["training"]["lr"],
        min_lr=config["training"]["min_lr"],
        wd=config["training"]["weight_decay"],
        loss=config["training"]["loss"],
        optimizer=config["training"]["optimizer"],
        head_dropout=config["model"]["head_dropout"],
        proj_dim=config["model"]["proj_dim"],
        head_depth=config["model"]["head_depth"],
        head_dim=config["model"]["head_dim"],
        pretrained=not args.randominit,
        activation_dim=config["model"]["activation_dim"],
        activation_nlayers=config["model"]["activation_nlayers"],
        activation_agglomeration=config["model"]["activation_agglomeration"],
        norm_embedding=config["model"]["norm_embedding"],
        head_norm=config["model"]["head_norm"],
        prenorm=config["model"]["prenorm"],
        linbranch=config["model"]["linbranch"],
        head_residual=config["model"]["head_residual"],
        learned_pooling=config["model"]["learned_pooling"],
        all_layers=config["model"]["all_layers"],
        ema_beta=config["training"]["ema_beta"],
        smoothness=config["training"]["smoothness"],
        diversity=config["training"]["diversity"],
        var=config["training"]["var"],
        orthogonality=config["training"]["orthogonality"],
        gradient_accumulation=config["training"]["gradient_accumulation"],
    )
    run_id = resultSocket.run_id if not OPTUNA else f"OPTUNA_{trial.number}"

    config["model"]["model_dir"] = f'{config["model"]["model_dir"]}/{run_id}' if not OPTUNA else config["model"]["model_dir"]

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
        head_dim = config["model"]["head_dim"] if config["model"]["head_dim"] > 0 else config["model"]["proj_dim"],
        head_depth = config["model"]["head_depth"],
        proj_dim = config["model"]["proj_dim"],
        use_clf_token = config["model"]["use_clf_token"],
        activation_dim=config["model"]["activation_dim"],
        activation_nlayers=config["model"]["activation_nlayers"],
        activation_agglomeration=config["model"]["activation_agglomeration"],
        norm_embedding=config["model"]["norm_embedding"],
        norm=config["model"]["head_norm"],
        prenorm=config["model"]["prenorm"],
        linbranch=config["model"]["linbranch"],
        head_residual=config["model"]["head_residual"],
        learned_pooling=config["model"]["learned_pooling"],
        all_layers=config["model"]["all_layers"],
    )
    if config["training"]["ema_beta"] != 0:
        ema_model = EMA(
            model,
            beta = config["training"]["ema_beta"],              # exponential moving average factor
            update_after_step = 500,    # only after this number of .update() calls will it start updating
            update_every = 10,
        )
    else:
        ema_model = None
    if not args.randominit:
        log("Loading pretrained weights")
        size = config["model"]["name"].split("_")[-1]
        utils.load_weights(model.backbone, size, config["model"]["weights_path"])
    model.to(device)
    ema_model and ema_model.to(device)
    if args.verbose >= 3:
        B = config["data"]["batch_size"]
        L = 100 # Max length of the sequence
        summary(model, input_data=(torch.randint(0, 20, (B, L))), device=device)
    log("Model loaded successfully!")

    # Loading optimizer, loss and scheduler
    if config["model"]["activation_dim"] > 0:
        log("Using Late Projection Loss")
        loss = LateProjCriterion(model.activation, config["training"]["loss"])
    else:
        log("Using cosine similarity loss")
        loss = Criterion(
            loss_type=config["training"]["loss"],
            smoothness=config["training"]["smoothness"],
            diversity=config["training"]["diversity"],
            var=config["training"]["var"],
            orthogonality=config["training"]["orthogonality"],
        )
        loss.to(device)
    optimizer = make_optimizer(model.parameters(),
                               loss.parameters() if config["model"]["activation_dim"] == 0 else [],
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
        ema_model=ema_model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss,
        num_epochs=config["training"]["num_epochs"],
        device=device,
        scheduler=scheduler,
        noscaler=args.noscaler,
        config=config,
        metrics=metrics,
        watch=args.watch,
        sample_inputs=sample_inputs,
        verbose=args.verbose,
        gradient_accumulation=config["training"]["gradient_accumulation"] or None
    )
    log("Training done!")

    # Load best model
    log("Loading best model")
    checkpoint = torch.load(f'{config["model"]["model_dir"]}/{config["model"]["name"]}.pth', weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    ema_model and log("Loading best ema model")
    ema_model and ema_model.load_state_dict(checkpoint["ema_state_dict"])
    # Test
    if OPTUNA:
        results, all_preds = evaluate(ema_model or model,
                                      val_loader, loss, device, metrics=metrics)
    else:
        results, all_preds = evaluate(ema_model or model,
                                      test_loader, loss, device, metrics=metrics)
        if config["data"]["dataset"] == "synt":
            test_rnd_align_loader = make_rnd_test_loader(config, alphabet)
            results_rnd, all_preds_rnd = evaluate(ema_model or model,
                                          test_rnd_align_loader, loss, device, metrics=metrics)
        log("Training done!  Saving...")

        # Prediction range
        plt.hist(all_preds.numpy(), bins=100, density=True)
        plt.title("Prediction distribution")
        plt.xlabel("Predicted Identity")
        plt.ylabel("Density")
        plt.grid()
        resultSocket.detect_and_log_figures(step=State.global_step, split="test", epoch=config["training"]["num_epochs"])
        plt.close()

        # Error
        error = test_loader.dataset.label - all_preds.numpy().squeeze()
        plt.hist(error, bins=100, density=True)
        plt.title("Error between true identity and predicted identity")
        plt.xlabel("Error (GT - pred)")
        plt.ylabel("Density")
        plt.grid()
        resultSocket.detect_and_log_figures(step=State.global_step, split="test", epoch=config["training"]["num_epochs"])
        plt.close()

        # Make the table
        abs_error = np.abs(error)
        labels = [0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantiles = [f"{value:.3f}" if abs(value) >= 0.001 else f"{value:.2e}"
                 for value in np.quantile(abs_error, labels)]
        rows = [[label, value] for label, value in zip(labels, quantiles)]
        html_table = utils.make_table(["Quantile", "Error"], rows)

        resultSocket.add_fragment(html_table, step=State.global_step, split="test", epoch=config["training"]["num_epochs"])

        save_dict = {
            "epoch": config["training"]["num_epochs"],
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "activation": loss.activation.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_results": results,
            "test_preds": all_preds.half()
        }
        torch.save(
            save_dict, f"{config['model']['model_dir']}/final_{config['model']['name']}.pth")
        # Copy config file to model dir
        shutil.copy(args.config, config['model']["model_dir"])

    # Print stats of code
    if config.have_warnings():
        warn(config.get_warnings())

    # Save results
    if config["data"]["dataset"] == "synt":
        resultSocket.write_result(**{name: result.item() for name, result in results.items()}, rnd_mae=results_rnd["mae"].item())
    else:
        resultSocket.write_result(**{name: result.item() for name, result in results.items()})

    return results[args.watch]


def experiment1_hsearch(args, kwargs):
    STUDY_NAME = "Experiment1"
    config = ConfigFile(args.config, config_format.get(option="ESM"), verify_path=True, profiles=["local", "hpc"])
    config.change_profile(utils.get_profile())
    config.override_config(kwargs)
    BASE_MODEL_DIR = config['model']['model_dir']

    def objective(trial: optuna.trial.Trial):
        # Sample parameters for the training
        config["model"]["model_dir"] = f'{BASE_MODEL_DIR}/OPTUNA_{trial.number}'

        # Training
        config["training"]["lr"] = trial.suggest_float('training.lr', 1e-6, 1e-3, log=True)
        config["training"]["weight_decay"] = trial.suggest_float('training.weight_decay', 1e-12, 1., log=True)

        return experiment1(args, kwargs, config, trial)

    # Define the study
    direction = "minimize" if args.watch == "loss" or args.watch == "mae" else "maximize"
    log(f"Optimizing in direction '{direction}'")
    db = f"hsearch_{STUDY_NAME}"
    study = optuna.create_study(direction=direction, study_name=f'{STUDY_NAME}',
                                storage=f"sqlite:///{db}",
                                load_if_exists=True, pruner=optuna.pruners.MedianPruner(
                                                                        n_startup_trials=5,  # Wait for 5 trials before pruning starts
                                                                        n_warmup_steps=5,   # Don't prune in first 10 steps of training
                                                                        interval_steps=1     # Check every epoch
                                                                    )
                                )
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # Pretty print the results
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    log("Search Done!")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save in results the best parameters
    savepath = f"{BASE_MODEL_DIR}/experiment1_hparams.yml"
    log(f"saving hyperparameters to {savepath}")
    with open(savepath, 'w') as outfile:
        yaml.dump(trial.params, outfile, default_flow_style=False)