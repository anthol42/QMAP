import torch
import torch.nn as nn
from utils.esm_alphabet import ESMAlphabet
from .esm import ESM


class FC_projector(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int, out_features: int, max_length: int,
                 n_encoder_layers: int):
        super().__init__()
        self.max_length = max_length
        self.n_encoder_layers = n_encoder_layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, latent_dim),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
            ),
            nn.Linear(latent_dim, out_features)
        ]
        )
        self.linbranch = nn.Linear(embed_dim, out_features)

        self.layer_weight = nn.Parameter(torch.ones(n_encoder_layers, 1, 1, 1) / n_encoder_layers)

    def forward(self, x, padding_mask=None):
        # x: shape(N, B, T, D) where N is the number of layers
        x = (self.layer_weight * x).sum(dim=0)

        if padding_mask is not None:
            value_mask = 1 - padding_mask.to(x.dtype).unsqueeze(-1)
            x = (x * value_mask).sum(dim=1) / value_mask.sum(dim=1)
        else:
            x = x.mean(dim=1)

        h = self.layers[0](x)  # Initial projection

        for layer in self.layers[1:-1]: # Intermediate layers with residuals (1 layer in our case)
            residual = h
            out = layer(h)
            h = out + residual

        out = self.layers[-1](h)  # Final proj

        linout = self.linbranch(x)
        return out + linout

class ESMEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 12,
        embed_dim: int = 480,
        attention_heads: int = 20,
        token_dropout: bool = True,
    ):
        super().__init__()
        self.backbone = ESM(
            alphabet=ESMAlphabet(),
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            token_dropout=token_dropout,
            attention_dropout=0.,
            layer_dropout=0.,

        )
        self.head = FC_projector(embed_dim, 512, 512, 102, num_layers)
        self.init_params = dict(
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            token_dropout=token_dropout
        )


    def forward(self, tokens):
        padding_mask = tokens.eq(self.backbone.padding_idx)
        z = self.backbone(tokens)  # Shape (B, T, E)

        emb = self.head(z, padding_mask=padding_mask)

        # Normalize the embeddings
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb
