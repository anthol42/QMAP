from typing import *
import torch
from utils.esm_alphabet import ESMAlphabet

class BaseCollator:
    def __init__(self, alphabet: ESMAlphabet, max_len: int = 100):
        self.alphabet = alphabet
        self.max_len = max_len

    def __call__(self, raw_batch: Sequence[Tuple[str, torch.Tensor]]) -> \
            Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Called by the dataloader to collate a batch of data.
        :param raw_batch: The raw batch of data
        :return: A list of the raw sequences, the tokenized ground truth, the tokenized input and the label
        """
        raise NotImplementedError("This method must be implemented in the child class")

    def tokenize_batch(self, X):
        return [self.alphabet.encode(seq) for seq in X] # Make tensors

    def prep_tensors(self, B, L):
        tokens = torch.empty((B, L + 2), dtype=torch.int64) # We add 1 because of the <bos> token, aka <cls>
        tokens.fill_(self.alphabet.padding_idx)

        return tokens