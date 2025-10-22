from pyutils import Default
config_format = {
    "data":{
        "batch_size": int,
        "shuffle": Default(bool, True),
        "path": str,
        "num_workers": int,
        "dataset": Default(str, "")
    },
    "training":{
        "num_epochs": int,
        "lr": float,
        "smoothness": Default(float, 0.),
        "diversity": Default(float, 0.),
        "var": Default(float, 0.),
        "orthogonality": Default(float, 0.),
    },
    "model":{
            "model_dir": str,
            "weights_path": str,
            "name": str,
            "num_layers": int,
            "embed_dim": int,
            "attention_heads": int,
            "token_dropout": Default(bool, True),
        }
}