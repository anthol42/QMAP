from .base_collator import BaseCollator
import torch
from typing import Sequence, Tuple, List

class AlignmentCollator(BaseCollator):
    def __call__(self, raw_batch: Sequence[Tuple[str, str, float]]) -> \
            Tuple[List[str], List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Called by the dataloader to collate a batch of data.
        :param raw_batch: The raw batch of data
        :return: A list of the raw sequences, the tokenized ground truth, the tokenized input and the label
        """
        B = len(raw_batch)
        seq1, seq2, label = zip(*raw_batch)
        L1 = max(len(s) for s in seq1)
        L2 = len(seq2)

        # Tokenize
        toks1 = self.tokenize_batch(seq1)
        toks2 = self.tokenize_batch(seq2)

        # Prepare the tensors
        tokens1 = self.prep_tensors(B, L1)
        tokens2 = self.prep_tensors(B, L2)
        targets = torch.tensor(label).unsqueeze(1).float()

        # Fill the tensors
        tokens1 = self.fill_tensors(tokens1, toks1, self.alphabet)
        tokens2 = self.fill_tensors(tokens2, toks2, self.alphabet)

        return seq1, seq2, tokens1, tokens2, targets

    @staticmethod
    def fill_tensors(tokens: torch.Tensor, all_toks: List[List[int]], alphabet) -> torch.Tensor:
        """
        Fill the tokens tensor with the tokenized sequences.
        :param tokens: The tokens tensor to fill
        :param all_toks: The tokenized sequences
        :param alphabet: The alphabet used for tokenization
        :return: The filled tokens tensor
        """
        for i, toks in enumerate(all_toks):
            tokens[i, 1:len(toks) + 1] = toks
            tokens[i, len(toks)] = alphabet.eos_idx
        tokens[:, 0] = alphabet.cls_idx
        return tokens