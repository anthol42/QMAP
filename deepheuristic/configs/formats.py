from pyutils import ConfigFormat, Option, Options, Default, Profile
config_format = ConfigFormat({
    "data":{
        "batch_size": int,
        "shuffle": bool,
        "path": Default(str, ".cache/")
    },
    "training":{
        "num_epochs": Profile(int),
        "learning_rate": float,
        "weight_decay": float,
    },
    "model":Options(
        Option("CNN")(
            {
            "model_dir": "opath",
            "name": "opath",
            "dropout2d": float,
            "dropout": float
            }
        ),
        Option("Transformer")({
            "not": str,
            "implemented": str
        })
    ),
    "scheduler":{
        "n_iter_restart": int,
        "factor_increase": int,
        "min_lr": float
    },
})