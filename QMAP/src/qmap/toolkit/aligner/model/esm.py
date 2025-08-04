import torch
import torch.nn as nn
from typing import Iterable, Optional, Literal
from pyutils import ConfigFile
from .esm_alphabet import ESMAlphabet
from .modules import ESM1bLayerNorm, TransformerLayer

class FC_layer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float,
                 norm: Literal['Batch', 'Layer', 'ESM', 'none'] = 'ESM',
                 prenorm: bool = False):
        super().__init__()
        if norm is None:
            norm = 'none'
        match norm:
            case 'Batch':
                norm = nn.BatchNorm1d
            case 'Layer':
                norm = nn.LayerNorm
            case 'ESM':
                norm = ESM1bLayerNorm
            case 'none':
                norm = nn.Identity
            case _:
                raise ValueError(f'Unknown norm: {norm}')

        if dropout == 0:
            if prenorm:
                self.layers = nn.Sequential(
                    norm(in_dim),
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                )
            else:
                self.layers = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    norm(out_dim),
                    nn.GELU(),
                )
        else:
            if prenorm:
                self.layers = nn.Sequential(
                    norm(in_dim),
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            else:
                self.layers = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    norm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
    def forward(self, x):
        return self.layers(x)


class FC_projector(nn.Module):
    def __init__(self, depth: int, embed_dim: int, latent_dim: int, out_features: int, max_length: int,
                 n_encoder_layers: int, dropout: float,
                 use_clf_token: bool = False,
                 norm: Literal['Batch', 'Layer', 'ESM', 'none'] = 'ESM',
                 prenorm: bool = False,
                 linbranch: bool = False,
                 residual: bool = False,
                 learned_pooling = False,
                 all_layers: bool = False,
                 ):
        super().__init__()
        self.max_length = max_length
        self.n_encoder_layers = n_encoder_layers
        self.all_layers = all_layers
        if depth == 0:
            self.layers = nn.ModuleList(
                [nn.Linear(embed_dim, out_features)]
            )
        elif depth == 1:
            self.layers = nn.ModuleList(
                [
                FC_layer(embed_dim, latent_dim, dropout=0., norm=norm, prenorm=prenorm),
                nn.Linear(latent_dim, out_features)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                FC_layer(embed_dim, latent_dim, dropout=dropout, norm=norm, prenorm=prenorm),
                *[FC_layer(latent_dim, latent_dim, dropout, norm=norm, prenorm=prenorm) for _ in range(depth - 2)],
                FC_layer(latent_dim, latent_dim, dropout=0., norm=norm, prenorm=prenorm),
                nn.Linear(latent_dim, out_features)
                 ]
            )
        self.use_clf_token = use_clf_token
        if linbranch:
            self.linbranch = nn.Linear(embed_dim, out_features)
        else:
            self.linbranch = None
        if learned_pooling:
            self.pooling_param = nn.Parameter(torch.ones(max_length, 1)) # Start as normal mean
        else:
            self.pooling_param = None

        if self.all_layers:
            self.layer_weight = nn.Parameter(torch.ones(n_encoder_layers, 1, 1, 1) / n_encoder_layers)
        else:
            self.layer_weight = None
        self.isres = residual
        self._init()

    def _init(self):
        pass
        # layer = self.layers[-1]
        # nn.init.xavier_uniform_(layer.weight, gain=0.5)
        # if layer.bias is not None:
        #     nn.init.constant_(layer.bias, 0.)

    def forward(self, x, padding_mask=None):
        # x: shape(N, B, T, D) where N is the number of layers
        L = x.shape[2]
        if self.all_layers:
            x = (self.layer_weight * x).sum(dim=0)
        else:
            x = x[-1]
        if self.use_clf_token:
            x = x[:, 0]
        else:
            if padding_mask is not None:
                value_mask = 1 - padding_mask.to(x.dtype).unsqueeze(-1)
                if self.pooling_param is not None:
                    x = self.pooling_param[:L] * x
                x = (x * value_mask).sum(dim=1) / value_mask.sum(dim=1)
            else:
                if self.pooling_param is not None:
                    x = self.pooling_param[:L] * x
                x = x.mean(dim=1)
        if self.isres:
            out = self.res_forward(x)
        else:
            out = self.sequential(x)
        if self.linbranch is not None:
            linout = self.linbranch(x)
            return out + linout
        else:
            return out

    def res_forward(self, x):
        h = self.layers[0](x) # Initial projection
        for layer in self.layers[1:-1]:
            residual = h
            out = layer(h)
            h = out + residual
        if len(self.layers) > 1:
            h = self.layers[-1](h) # Final proj
        return h

    def sequential(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ESM(nn.Module):
    def __init__(
        self,
        alphabet: ESMAlphabet,
        num_layers: int,
        embed_dim: int,
        attention_heads: int,
        token_dropout: bool,
        attention_dropout: float,
        layer_dropout: float
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

        all_activations = []
        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                self_attn_padding_mask=padding_mask,
            ) # Shape(T, B, E)
            all_activations.append(x.transpose(0, 1)) # (T, B, E) => (B, T, E)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        all_activations[-1] = x

        return torch.stack(all_activations, dim=0)

class Activation(nn.Module):
    def __init__(self, proj_dim: int, n_layers: int, agglomeration_type: Literal['mult', 'abs_diff', 'cat'] = 'mult'):
        """
        Module used to compute the identity between two sequence from their vector representations.
        """
        super().__init__()
        self.agglomeration = agglomeration_type

        if n_layers < 1:
            raise ValueError('n_layers must be >= 1')

        if agglomeration_type == 'cat':
            proj_dim *= 2

        if n_layers == 1:
            self.layers = nn.Sequential(
                nn.Linear(proj_dim, 2 * proj_dim),
                nn.GELU(),
                nn.Linear(2 * proj_dim, 1),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(proj_dim, 2 * proj_dim),
                nn.GELU(),
                *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(2 * proj_dim, 2 * proj_dim), nn.GELU())],
                nn.Linear(2 * proj_dim, 1),
            )

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor):
        """
        Compute the pseudo-identity between two embeddings.
        :param emb1: The first embedding tensor of shape (B, E)
        :param emb2: The second embedding tensor of shape (B, E)
        :return: (B, 1) tensor with the pseudo-identity (between 0 and 1)
        """
        if self.agglomeration == 'mult':
            dist = emb1 * emb2 # Shape (B, 2E)
        elif self.agglomeration == 'abs_diff':
            dist = (emb1 - emb2).abs()
        elif self.agglomeration == 'cat':
            dist = torch.cat((emb1, emb2), dim=-1)
        else:
            raise NotImplementedError(f"Unknown agglomeration type: {self.agglomeration}")
        out = self.layers(dist) # dist @ self.weight + self.bias  # Shape (B, 1)
        return out # To get the pseudo-identity, we need to apply a sigmoid activation


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
        norm_embedding: bool = True,
        norm: Literal['Batch', 'Layer', 'ESM', 'none'] = 'ESM',
        prenorm: bool = False,
        linbranch: bool = False,
        head_residual: bool = False,
        learned_pooling: bool = False,
        all_layers: bool = False
    ):
        super().__init__()
        self.backbone = ESM(
            alphabet=alphabet,
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            token_dropout=token_dropout,
            attention_dropout=attention_dropout,
            layer_dropout=layer_dropout,

        )
        self.head = FC_projector(head_depth, embed_dim, head_dim, proj_dim, 102, num_layers,
                                      head_dropout, use_clf_token=use_clf_token, norm=norm,
                                 prenorm=prenorm, linbranch=linbranch, residual=head_residual,
                                 learned_pooling=learned_pooling, all_layers=all_layers)
        self.init_params = dict(
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            token_dropout=token_dropout,
            attention_dropout=attention_dropout,
            layer_dropout=layer_dropout,
            head_dropout=head_dropout,
            head_dim=head_dim,
            head_depth=head_depth,
            proj_dim=proj_dim,
            norm_embedding=norm_embedding,
            norm=norm,
            prenorm=prenorm,
            linbranch=linbranch,
            head_residual=head_residual,
        )
        self.activation = nn.PReLU()

        self.norm_emb = norm_embedding

    def forward(self, tokens):
        padding_mask = tokens.eq(self.backbone.padding_idx)
        z = self.backbone(tokens)  # Shape (B, T, E)

        emb = self.head(z, padding_mask=padding_mask)

        # Normalize the embeddings
        if self.norm_emb:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    @classmethod
    def Load(cls, frm: str):
        data = torch.load(frm)
        params = data['params']
        params["alphabet"] = ESMAlphabet()  # Ensure alphabet is set
        model = cls(**params)
        model.load_state_dict(data['state_dict'])
        return model


    def save(self, to: str):
        data = {
            'state_dict': self.state_dict(),
            'params': self.init_params
        }

        torch.save(data, to)
