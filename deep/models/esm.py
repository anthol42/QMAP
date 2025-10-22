import torch
import torch.nn as nn
from utils.esm_alphabet import ESMAlphabet
from .modules import ESM1bLayerNorm, TransformerLayer

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