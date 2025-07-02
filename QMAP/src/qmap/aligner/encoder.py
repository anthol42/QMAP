from .model import ESMEncoder
from .model import ESMAlphabet
import torch
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple, Sequence, Literal
from pyutils import progress
from .vectorizedDB import VectorizedDB
from .utils import _get_device

root = Path(__file__).parent.parent

@dataclass
class EsmEncoderConfig:
    num_layers: int = 12
    embed_dim: int = 480
    attention_heads: int = 20
    token_dropout: bool = True
    attention_dropout: float = 0.
    layer_dropout: float = 0.
    head_dropout: float = 0.
    head_dim: int = 512
    head_depth: int = 0
    proj_dim: int = 512
    use_clf_token: bool = True


class AlignmentCollator:
    def __init__(self, alphabet: ESMAlphabet, max_len: int = 100):
        self.alphabet = alphabet
        self.max_len = max_len

    def __call__(self, raw_batch: Sequence[Tuple[str, str, float]]) -> \
            Tuple[List[str], torch.Tensor]:
        """
        Called by the dataloader to collate a batch of data.
        :param raw_batch: The raw batch of data
        :return: A list of the raw sequences, the tokenized ground truth, the tokenized input and the label
        """
        B = len(raw_batch)
        seq = raw_batch
        L = max(len(s) for s in seq)
        if L > self.max_len:
            print(seq)
            raise ValueError(f"Sequence length {L} exceeds maximum length {self.max_len}. "
                             f"Please increase the max_len parameter in the collator.")

        # Tokenize
        toks = self.tokenize_batch(seq)

        # Prepare the tensors
        tokens = self.prep_tensors(B, L)

        # Fill the tensors
        tokens = self.fill_tensors(tokens, toks, self.alphabet)

        return seq, tokens

    def tokenize_batch(self, X):
        return [self.alphabet.encode(seq) for seq in X] # Make tensors

    def prep_tensors(self, B, L):
        tokens = torch.empty((B, L + 2), dtype=torch.int64) # We add 1 because of the <bos> token, aka <cls>
        tokens.fill_(self.alphabet.padding_idx)

        return tokens

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

class SeqDataset(Dataset):
    def __init__(self, sequences: List[Tuple[int, str]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx][1]

class Encoder:
    def __init__(self, config: EsmEncoderConfig = EsmEncoderConfig(),
                 path: str = f"{root}/aligner/model/weights/ESM_35M.pth",
                 force_cpu: bool = False):
        self.device = _get_device(force_cpu=force_cpu)

        self.alphabet = ESMAlphabet()
        self.model = ESMEncoder(
            alphabet=self.alphabet,
            num_layers=config.num_layers,
            embed_dim=config.embed_dim,
            attention_heads=config.attention_heads,
            token_dropout=config.token_dropout,
            attention_dropout=config.attention_dropout,
            layer_dropout=config.layer_dropout,
            head_dropout=config.head_dropout,
            head_dim=config.head_dim,
            head_depth=config.head_depth,
            proj_dim=config.proj_dim,
            use_clf_token=config.use_clf_token
        )
        data = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(data["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sequences: list[str]) -> VectorizedDB:
        """
        Encode a list of sequences using the ESM model.
        :param sequences: List of protein sequences to encode.
        :return: Encoded tensor of shape (batch_size, sequence_length, embed_dim).
        """
        # Make dataloader
        dataloader = self._make_dataloader(sequences, self.alphabet, batch_size=64)

        all_embeddings = []
        all_sequences = []
        with torch.inference_mode():
            for seqs, tokens in progress(dataloader, type="pip", desc="Encoding sequences", display=len(dataloader) > 1):
                tokens = tokens.to(self.device)
                embeddings = self.model(tokens)
                all_embeddings.append(embeddings.cpu().half())
                all_sequences += seqs

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Rever back the embeddings to the original order
        indices = [i for i, _ in dataloader.dataset.sequences]
        sorting_indices = torch.argsort(torch.tensor(indices))
        all_embeddings = all_embeddings[sorting_indices]
        all_sequences = [all_sequences[i] for i in sorting_indices]

        # Make the Vectorize Sequence Db object
        return VectorizedDB(all_sequences, all_embeddings)


    @staticmethod
    def _make_dataloader(sequences: list[str],
                         alphabet: ESMAlphabet,
                         batch_size: int) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for the sequences.
        :param sequences: List of protein sequences to encode.
        :param alphabet: The alphabet used for tokenization.
        :param batch_size: The size of each batch.
        :return: DataLoader for the sequences.
        """
        sequences = [(i, seq) for i, seq in enumerate(sequences)]
        sorted_sequences = sorted(sequences, key=lambda x: len(x[1]))
        collator = AlignmentCollator(alphabet=alphabet)
        dataloader = torch.utils.data.DataLoader(
            SeqDataset(sorted_sequences),
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
        )
        return dataloader