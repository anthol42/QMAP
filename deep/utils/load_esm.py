from typing import Literal
import os
import esm
import torch
from pathlib import PurePath
from glob import glob

def clean_keys(state_dict, forbidden_prefix=('contact_head', 'lm_head')):
    """Clean the keys of a state dict to remove 'module.' prefix."""
    out = {}
    for key in state_dict.keys():
        if not any(key.startswith(prefix) for prefix in forbidden_prefix):
            out[key] = state_dict[key]

    return out

def get_esm_weights(size: Literal["8M", "35M", "150M", "650M"], path: str):
    """
    Load the ESM weights from the given path if available, otherwise download them.
    :param size: The size of the ESM model to load. Options are "8M", "35M", "150M", "650M".
    :param path: The path where the ESM model weights are stored or will be downloaded to.
    :return: The state_dict of the ESM model.
    """
    if size not in ("8M", "35M", "150M", "650M"):
        raise ValueError(f"Invalid size '{size}'. Must be one of '8M', '35M', '150M', or '650M'.")
    if not os.path.exists(path):
        os.makedirs(path)
    path = PurePath(path) / f"{size}.pth"
    if os.path.exists(path):
        print(f"Loading ESM {size} weights from {path}")
        return torch.load(path)
    else:
        print(f"Downloading ESM {size} weights to {path}")
        match size:
            case "8M":
                esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            case "35M":
                esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            case "150M":
                esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            case "650M":
                esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            case _:
                raise ValueError(f"Invalid size '{size}'")

        esm_model.eval()
        esm_model.to("cpu")  # Ensure the model is on CPU for saving
        state_dict = clean_keys(esm_model.state_dict())

        torch.save(state_dict, path)
        # Now, delete the version saved to torch path
        torch_path = os.environ.get("TORCH_HOME")
        if torch_path is None:
            torch_path = os.environ.get("XDG_CACHE_HOME")
        if torch_path is None:
            torch_path = os.path.join(os.path.expanduser("~"), ".cache", "torch")
        hub_path = PurePath(torch_path) / "hub" / "checkpoints"
        esm_paths = glob(str(hub_path / f"esm2_*"))
        for esm_path in esm_paths:
            os.remove(esm_path)
        return state_dict

def load_weights(model: torch.nn.Module, size: Literal["8M", "35M", "150M", "650M"], path: str):
    state_dict = get_esm_weights(size, path)
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if key.endswith("cos_cached") and key not in state_dict:
            state_dict[key] = model_state_dict[key]
        elif key.endswith("sin_cached") and key not in state_dict:
            state_dict[key] = model_state_dict[key]

    model.load_state_dict(state_dict)