from .model import ESMEncoder
from .model import ESMAlphabet
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Tuple, Sequence, Optional
from pyutils import progress
from .vectorizedDB import VectorizedDB
from .utils import _get_device
from huggingface_hub import PyTorchModelHubMixin

root = Path(__file__).parent.parent

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

class QMAPModel(
    torch.nn.Module,
    PyTorchModelHubMixin,
    repo_url="anthol42/qmap"
):
    def __init__(self, config):
        super().__init__()
        alphabet = ESMAlphabet()
        self.encoder = ESMEncoder(
            alphabet=alphabet,
            **config
        )
        self.activation = torch.nn.PReLU()


    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        :param seqs: Tensor of shape (batch_size, sequence_length) containing token indices.
        :return: Tensor of shape (batch_size, embed_dim) containing embeddings.
        """
        return self.encoder(seqs)


class Encoder:
    """
    This class is used to encode the protein sequences into embeddings. It will return a VectorizedDB object which is
    mainly a wrapper of a tensor of shape (N_sequences, 512) corresponding to the sequence embeddings.

    Usage:

    Its usage is very simple, simply initialize the class and call the encode method with a list of sequences!
    """
    def __init__(self, force_cpu: bool = False):
        self.device = _get_device(force_cpu=force_cpu)

        model = QMAPModel.from_pretrained('anthol42/qmap')
        model.to(self.device)
        model.eval()

        self.model = model.encoder
        self.activation = model.activation
        self.alphabet = ESMAlphabet()

    def encode(self, sequences: list[str], batch_size: int = 512, ids: Optional[List[str]] = None) -> VectorizedDB:
        """
        Encode a list of sequences using the model.
        :param sequences: List of peptide sequences to encode. Note that the sequences should have a maximum length of 100 amino acids. Please filter out longer sequences or truncate them before encoding.
        :param ids: The sequence ids. Useful when the sequences are not unique. If not provided, you can still find the sequence by its index as the order of the embeddings is the same as the order of the sequences.
        :param batch_size: The batch size to use. Change this to a lower value if you run out of memory.
        :return: Encoded tensor of shape (N_sequences, 512).
        """
        # Make dataloader
        dataloader = self._make_dataloader(sequences, self.alphabet, batch_size=batch_size)

        all_embeddings = []
        all_sequences = []
        with torch.inference_mode():
            self.model.eval()
            for seqs, tokens in progress(dataloader, type="pip", desc="Encoding sequences", display=len(dataloader) > 1):
                tokens = tokens.to(self.device)
                embeddings = self.model(tokens).cpu()
                all_embeddings.append(embeddings.half())
                all_sequences += seqs

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Rever back the embeddings to the original order
        indices = [i for i, _ in dataloader.dataset.sequences]

        sorting_indices = torch.argsort(torch.tensor(indices))

        all_embeddings = all_embeddings[sorting_indices]
        all_sequences = [all_sequences[i] for i in sorting_indices]

        # Make the Vectorize Sequence Db object
        return VectorizedDB(all_sequences, all_embeddings, ids=ids)


    @staticmethod
    def _make_dataloader(sequences: list[str],
                         alphabet: ESMAlphabet,
                         batch_size: int) -> DataLoader:
        """
        Create a DataLoader for the sequences.
        :param sequences: List of protein sequences to encode.
        :param alphabet: The alphabet used for tokenization.
        :param batch_size: The size of each batch.
        :return: DataLoader for the sequences.
        """
        sequences = [(i, seq) for i, seq in enumerate(sequences)]
        max_len = max(len(seq) for _, seq in sequences)
        if  max_len > 100:
            raise ValueError(f"Some sequences are too long ({max_len} > 100). ")
        sorted_sequences = sorted(sequences, key=lambda x: len(x[1]))
        collator = AlignmentCollator(alphabet=alphabet)
        dataloader = DataLoader(
            SeqDataset(sorted_sequences),
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
        )
        return dataloader