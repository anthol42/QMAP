from torch.utils.data import Dataset
from pyutils import ConfigFile
from utils.esm_alphabet import ESMAlphabet
from typing import Literal
import utils
import numpy as np

class AlignmentDataset(Dataset):
    def __init__(self, config: ConfigFile, split: Literal["train", "val", "test"]):
        self.config = config
        self.sequences = self._load_sequences(config["data"]["path"], split)
        self.seq_pair, self.label = self._load_annotations(config["data"]["path"], split) # Alignment identities
        self.split = split
        self.max_length = 100


    def __len__(self):
        return len(self.seq_pair)

    def __getitem__(self, idx):
        """
        Sample a sequence pair and its label by index.
        :param idx: The index of the sequence pair to sample.
        :return: Sequence 1, Sequence 2, Label
        """
        id1, id2 = self.seq_pair[idx]
        label = self.label[idx]
        seq1 = self.sequences[id1]
        seq2 = self.sequences[id2]

        return seq1, seq2, label


    def _load_sequences(self, path: str, split: Literal["train", "val", "test"]):
        """
        Load sequences from fasta file from the specified path and split.
        :param path: Path to the build directory Containing the fasta and npy files
        :param split: The split to load (train, val, test)
        :return: A dictionary mapping ids to sequences
        """
        fasta_path = f"{path}/{split}.fasta"
        return utils.read_fasta(fasta_path)

    def _load_annotations(self, path: str, split: Literal["train", "val", "test"]):
        """
        Load the annotation for a given split. It returns the sequences pairs as an int array and the annotations as a
        float array
        :param path: The path to the build directory containing the npy files
        :param split: The split to load (train, val, test)
        :return: The sequence pair ids and the labels (identities)
        """
        npy_path = f"{path}/{split}.npy"
        data = np.load(npy_path)
        seq_pair = data[:, :2].astype(np.int32)
        label = data[:, 2].astype(np.float32)
        return seq_pair, label
