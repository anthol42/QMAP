# Deeplearning based model to approximate identity calculation between two peptides

This directory contains the code to train and evaluate our deep learning model. It is implemented as a CLI application.

## How to run
1. First, sync the uv env
```shell
uv sync
```
2. Then, modify the config file `config/ESM_35M.yml` so that it points to the path of where you stored the data.
- Change the `data.path` field to point to the directory where you stored the data.
- Change the `model.model_dir` field to point to where you want to checkpoints to be saved.
- Change the `weights_path` field to point to where you want the official ESM2 weights to be stored.
3. Run the training by running:
```shell
uv run main.py configs/ESM_35M.yml
```


## Code structure
The code is structured in different modules. The root of the code can be found in the `experiments/experiment1.py` file.
This is the file that is run by the CLI application.

The `data/` contains the files to load the data and format it to be fed to the model.  

The `models/` contains the model definition. The `models/esm_encoder` is the module that contains our encoder 
implementation that depends on the esm model. The other files are related to the ESM model, cloned from the official 
repo.

The `training/` contains the training loop and evaluation code.

The `loss.py` contains the loss function definition.

Finally, the `test_qmap_encoder.py` file contains a script that evaluates the performances of a checkpoint, and push the hub.

