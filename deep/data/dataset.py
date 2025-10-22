from torch.utils.data import Dataset
from pyutils import ConfigFile
from typing import Literal
import utils
import numpy as np

class AlignmentDataset(Dataset):
    def __init__(self, config: ConfigFile, split: Literal["train", "val", "test"], fract: float = 1.):
        self.config = config
        self.sequences = self._load_sequences(config["data"]["path"], split)
        self.seq_pair, self.label = self._load_annotations(config["data"]["path"], split, fract) # Alignment identities
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

    @staticmethod
    def _load_sequences(path: str, split: Literal["train", "val", "test"]):
        """
        Load sequences from fasta file from the specified path and split.
        :param path: Path to the build directory Containing the fasta and npy files
        :param split: The split to load (train, val, test)
        :return: A dictionary mapping ids to sequences
        """
        fasta_path = f"{path}/{split}.fasta"
        return utils.read_fasta(fasta_path)

    @staticmethod
    def _load_annotations(path: str, split: Literal["train", "val", "test"], fract: float):
        """
        Load the annotation for a given split. It returns the sequences pairs as an int array and the annotations as a
        float array
        :param path: The path to the build directory containing the npy files
        :param split: The split to load (train, val, test)
        :param fract: The fraction of the sequences to load
        :return: The sequence pair ids and the labels (identities)
        """
        npy_path = f"{path}/{split}.npy"
        data = np.load(npy_path)
        seq_pair = data[:, :2].astype(np.int32)
        label = data[:, 2].astype(np.float32)

        if fract < 1.:
            # Load the fraction of the data
            indices = np.arange(seq_pair.shape[0])
            np.random.shuffle(indices)
            indices = indices[:int(len(indices)*fract)]
            seq_pair = seq_pair[indices]
            label = label[indices]
        return seq_pair, label
