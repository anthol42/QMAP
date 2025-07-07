from pyutils import ConfigFormat, Option, Options, Default, Profile
config_format = ConfigFormat({
    "data":{
        "batch_size": Profile(int),
        "shuffle": Default(bool, True),
        "path": Profile(str),
        "num_workers": Profile(int),
    },
    "training":{
        "num_epochs": int,
        "lr": float,
        "min_lr": float,
        "weight_decay": float,
        "loss": str, # MSE or BCE
        "optimizer": str, # Adam or AdamW
        "ema_beta": Default(float, 0.)
    },
    "model":Options(
        Option("ESM")({
            "model_dir": Profile(str),
            "weights_path": Profile(str),
            "name": str,
            "num_layers": int,
            "embed_dim": int,
            "attention_heads": int,
            "token_dropout": Default(bool, True),
            "attention_dropout": Default(float, 0.),
            "layer_dropout": Default(float, 0.),
            "head_dropout": float,
            "head_depth": Default(int, 0), # We default to a Linear projection
            "head_dim": Default(int, 0), # By default, it is the proj dim (if 0)
            "proj_dim": int,
            "use_clf_token": Default(bool, True),
            "activation_dim": Default(int, 0),
            "activation_nlayers": Default(int, 1),
            "activation_agglomeration": Default(str, "mult"),
            "norm_embedding": Default(bool, True),
            "head_norm": Default(str, "ESM"),
            "prenorm": Default(bool, False),
            "linbranch": Default(bool, False),
            "head_residual": Default(bool, False),
            "learned_pooling": Default(bool, False),
            "all_layers": Default(bool, False),
        })
    ),
})