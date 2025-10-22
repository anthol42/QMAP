import torch
from typing import *


class ESMAlphabet:
    """
    ESM2 compatible alphabet.
    """
    def __init__(self):
        self.standard_toks = list("LAGVSERTIDPKQNFYMHWCXBUZO.-")
        self.special_tokens = ['<cls>', '<pad>', '<eos>', '<unk>']
        self.all_toks = self.special_tokens + self.standard_toks # + ["<mask>"]
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(['<mask>'])
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")

    def get_idx(self, s: str):
        """
        Converty a string token to an index.
        :param s: The string token
        :return: The associated int token
        """
        return self.tok_to_idx.get(s, self.unk_idx)

    def get_s(self, tok: int):
        """
        Convert an index token to a string.
        :param tok: The int token
        :return: The string token
        """
        return self.all_toks[tok]
    def __len__(self):
        return len(self.all_toks)

    def encode(self, seq: str) -> torch.Tensor:
        """
        Encode a sequence into a tensor.
        :param seq: The sequence
        :return: The tensor
        """
        return torch.tensor([self.get_idx(c) for c in seq])

    def list_decode(self, tokens: torch.Tensor) -> List[str]:
        """
        Decode a tensor to a list of string
        :param tokens: The tokens to decode
        :return: The list
        """
        return [self.get_s(i) for i in tokens]


    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode a tensor into a sequence.
        :param tensor: The tensor
        :return: The sequence
        """
        return ''.join(self.list_decode(tokens))