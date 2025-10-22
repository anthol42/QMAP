"""
Test QMAP encoder, then push it to the Hugging Face Hub.
"""
import os

import torch
from models import ESMEncoder
from utils.esm_alphabet import ESMAlphabet
from ema_pytorch import EMA
from huggingface_hub import PyTorchModelHubMixin
import shutil
from loss import Criterion
from training.train import evaluate
from data import make_dataloader
import utils
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef
import dotenv

dotenv.load_dotenv()

metrics = {
    "mae": MeanAbsoluteError(),
    "pcc": PearsonCorrCoef(),
}

class QMAPModel(
    torch.nn.Module,
    PyTorchModelHubMixin,
    repo_url="anthol42/qmap"
):
    def __init__(self, config = None):
        super().__init__()
        config = {} if config is None else config
        model = ESMEncoder(**config)
        ema_model = EMA(model)
        checkpoint = torch.load(".weights/ESM_35M.pth", weights_only=False, map_location="cpu")
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
        self.encoder = ema_model.ema_model
        self.activation = torch.nn.PReLU()
        self.activation.load_state_dict(checkpoint["activation"])


    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        :param seqs: Tensor of shape (batch_size, sequence_length) containing token indices.
        :return: Tensor of shape (batch_size, embed_dim) containing embeddings.
        """
        return self.encoder(seqs)

if __name__ == "__main__":
    model = QMAPModel()

    model.save_pretrained("model_assets")

    # Make model card
    shutil.copyfile("model_assets/model_card.md", "model_assets/README.md")

    # Evaluate
    device = utils.get_device()
    alphabet = ESMAlphabet()
    # Loading the data
    data_cfg = {
        "data": {
            "path": "../peptide_atlas/build",
            "dataset": 'synt',  # Use None for default dataset
            "num_workers": 2,
            "batch_size": 512,
            "shuffle": True
        }
    }
    train_loader, val_loader, test_loader = make_dataloader(config=data_cfg, fract=1.)
    model = QMAPModel.from_pretrained("model_assets")
    model.to(device)

    loss = Criterion(
        smoothness=0.01,
        diversity=0.001,
        var=0.,
        orthogonality=0.05,
    )
    loss.activation = model.activation
    loss.to(device)

    results, all_preds = evaluate(model,
                                  test_loader, loss, device, metrics=metrics)
    print(results)

    # If everything passes, upload
    model.push_to_hub("anthol42/qmap", token=os.environ["HUGGING_FACE"])
