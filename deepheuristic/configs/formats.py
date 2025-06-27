from pyutils import ConfigFormat, Option, Options, Default, Profile
config_format = ConfigFormat({
    "data":{
        "batch_size": int,
        "shuffle": Default(bool, True),
        "path": Default(str, "../peptide_atlas/build"),
        "num_workers": Profile(int),
    },
    "training":{
        "num_epochs": int,
        "lr": float,
        "min_lr": float,
        "weight_decay": float,
        "loss": str, # MSE or BCE
        "optimizer": str, # Adam or AdamW
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
            "head_dim": int,
            "head_depth": int,
            "proj_dim": int,
            "use_clf_token": Default(bool, True),
            "sigmoid": bool # True if loss is MSE, False if BCE because we will use BCE with logits
        })
    ),
})