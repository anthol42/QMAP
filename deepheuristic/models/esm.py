import torch
import torch.nn as nn
from typing import Iterable, Optional
from pyutils import ConfigFile
from utils.esm_alphabet import ESMAlphabet
from .modules import ESM1bLayerNorm, TransformerLayer
import torch.nn.functional as F

class DropScaler(nn.Module):
    def __init__(self, ratio: float = 0.5):
        super().__init__()
        self.mult = 1 - ratio
    def forward(self, X):
        if self.training:
            return X * (self.mult ** -1)
        else:
            return X
class FC_layer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        if dropout == 0:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                ESM1bLayerNorm(out_dim),
                nn.GELU(),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                ESM1bLayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
    def forward(self, x):
        return self.layers(x)


class FC_projector(nn.Module):
    def __init__(self, depth: int, embed_dim: int, latent_dim: int, out_features: int, dropout: float,
                 use_clf_token: bool = False, sigmoid: bool = True):
        super().__init__()
        if depth == 0:
            self.layers = nn.Sequential(
                nn.Linear(embed_dim, out_features)
            )
        elif depth == 1:
            self.layers = nn.Sequential(
                FC_layer(embed_dim, latent_dim, dropout=0.),
                nn.Linear(latent_dim, out_features)
            )
        else:
            self.layers = nn.Sequential(
                FC_layer(embed_dim, latent_dim, dropout=dropout),
                *[FC_layer(latent_dim, latent_dim, dropout) for _ in range(depth - 2)],
                FC_layer(latent_dim, latent_dim, dropout=0.),
                nn.Linear(latent_dim, out_features),
            )
        self.use_clf_token = use_clf_token
        self.sigmoid = sigmoid

    def forward(self, x):
        # x: shape(B, T, D)
        if self.use_clf_token:
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(dim=1)
        out = self.layers(x)
        if self.sigmoid:
            return torch.sigmoid(out)
        else:
            return out

class ESMEncoder(nn.Module):
    def __init__(
        self,
        alphabet: ESMAlphabet = ESMAlphabet(),
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        token_dropout: bool = True,
        attention_dropout: float = 0.,
        layer_dropout: float = 0.,
        head_dropout: float = 0.1,
        head_dim: int = 1280,
        head_depth: int = 0, # 0 = Linear projection
        proj_dim: int = 8,
        use_clf_token: bool = False,
        sigmoid: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.padding_idx = self.alphabet.padding_idx
        self.mask_idx = self.alphabet.mask_idx
        self.cls_idx = self.alphabet.cls_idx
        self.eos_idx = self.alphabet.eos_idx
        self.token_dropout = token_dropout
        self.attention_dropout = attention_dropout
        self.layer_dropout = layer_dropout
        self.head_dim = head_dim
        self.head_depth = head_depth
        self.head_dropout = head_dropout
        self.proj_dim = proj_dim
        self.use_clf_token = use_clf_token
        self.sigmoid = sigmoid
        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                    attention_dropout=self.attention_dropout,
                    layer_dropout=self.layer_dropout
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)
        self.head = FC_projector(self.head_depth, self.embed_dim, self.head_dim, self.proj_dim,
                                      self.head_dropout, use_clf_token=self.use_clf_token, sigmoid=self.sigmoid)
    def forward(self, tokens):
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]    # Why do this? To normalized the distribution of token embedding to reflect the training distribution

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))


        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                self_attn_padding_mask=padding_mask,
            )

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        emb = self.head(x)
        return emb

    @classmethod
    def load(cls, frm: str):
        data = torch.load(frm)
        proj_keys = {k for k in data['state_dict'].keys() if k.startswith("projector")}

        if len(proj_keys) == 0:
            print("Projector is not in the weights.  This means that its weights will be initialized randomly")
            model = cls(**data['params'])
            model.load_state_dict(data['state_dict'], strict=False)
        else:
            print("Loading custom weights: projector is in the weights")
            model = cls(**data['params'])
            model.load_state_dict(data['state_dict'])

        return model


    def save(self, to: str):
        data = {
            'state_dict': self.state_dict(),
            'params': dict(num_layers=self.num_layers,
                           embed_dim=self.embed_dim,
                           attention_heads=self.attention_heads,
                           token_dropout=self.token_dropout)
        }

        torch.save(data, to)

class TransformerFreezer:
    def __init__(self, modules: Iterable[TransformerLayer]):
        self.modules = modules

    def _get_num_layers(self, ratio: float):
        return int(len(self.modules) * ratio)
    def freeze_attention_norm(self, ratio: float = 1.):
        layers_to_freeze = self._get_num_layers(ratio)
        for i in range(layers_to_freeze):
            self.modules[i].self_attn_layer_norm.requires_grad_(False)

    def freeze_final_norm(self, ratio: float = 1.):
        layers_to_freeze = self._get_num_layers(ratio)
        for i in range(layers_to_freeze):
            self.modules[i].final_layer_norm.requires_grad_(False)

    def freeze_ffn(self, ratio: float = 1.):
        layers_to_freeze = self._get_num_layers(ratio)
        for i in range(layers_to_freeze):
            self.modules[i].fc1.requires_grad_(False)
            self.modules[i].fc2.requires_grad_(False)

    def freeze_attention(self, ratio: float = 1.):
        layers_to_freeze = self._get_num_layers(ratio)
        for i in range(layers_to_freeze):
            self.modules[i].self_attn.requires_grad_(False)

class ESMFreezer:
    """
    Class that freezes specific layers of the ESM model
    """

    def __init__(self, model: ESMEncoder):
        self.model = model

    def _get_num_layers(self, ratio: float):
        return int(len(self.model.layers) * ratio)

    def freeze_embeddings(self):
        self.model.embed_tokens.requires_grad_(False)

    def freeze_transformer(self, ratio: float = 1.):
        layers_to_freeze = self._get_num_layers(ratio)
        for i in range(layers_to_freeze):
            self.model.layers[i].requires_grad_(False)

    def freeze_layernorm_after(self):
        self.model.emb_layer_norm_after.requires_grad_(False)

    @property
    def Transformer(self):
        return TransformerFreezer(self.model.layers)

def freeze_layers(model: ESMEncoder, config: ConfigFile):
    freezer = ESMFreezer(model)

    freezer.freeze_embeddings() if config["freezer"]["freeze_embeddings"] else None
    # if config["training"]["lora_rank"] > 0:
    #      freezer.freeze_transformer()
    if config["freezer"]["freeze_transformer"] > 0:
        freezer.freeze_transformer(config["freezer"]["freeze_transformer"])
    freezer.freeze_layernorm_after() if config["freezer"]["freeze_layernorm_after"] else None

    freezer.Transformer.freeze_attention_norm(config["freezer"]["freeze_attention_norm"]) if config["freezer"]["freeze_attention_norm"] else None
    freezer.Transformer.freeze_final_norm(config["freezer"]["freeze_final_norm"]) if config["freezer"]["freeze_final_norm"] else None
    freezer.Transformer.freeze_ffn(config["freezer"]["freeze_ffn"]) if config["freezer"]["freeze_ffn"] else None
    freezer.Transformer.freeze_attention(config["freezer"]["freeze_attention"]) if config["freezer"]["freeze_attention"] else None