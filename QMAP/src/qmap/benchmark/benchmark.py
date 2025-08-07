import numpy as np
from torch.utils.data import Dataset
from typing import List, Literal, Sequence, Optional
from .sample import Sample
import json

class QMAPBenchmark(Dataset):
    """
    Class representing the QMAP benchmark testing dataset. It is a subclass of `torch.utils.data.Dataset`, so it can be
    easily used with PyTorch's DataLoader. However, it is easy to extract the sequences and the labels from it to use
    it with other libraries such as tensorflow or keras.
    """
    def __init__(self,
                 modified_termini: bool = False,
                 unusual_aa: bool = False,
                 d_amino_acids: bool = False,
                 specie_as_input: bool = False,
                 species_subset: Optional[List[str]] = None,
                 dataset_type: Literal['MIC', 'Hemolytic', 'Cytotoxic'] = 'MIC',
                 show_all: bool = False
                 ):
        """
        :param modified_termini: If True, the dataset will return the N and C terminus smiles string. Otherwise, sequences containing modified termini will be skipped.
        :param unusual_aa: If True, the dataset will return the unusual amino acids as a dictionary of positions and their. Otherwise, sequences containing unusual amino acids will be skipped.
        :param d_amino_acids: If True, the dataset will return sequences with D-amino acids. Otherwise, sequences containing D-amino acids will be skipped.
        :param specie_as_input: If True, the dataset will return a tuple of (sequence, specie) and the target will be a scalar. Otherwise, all species in species_subset will be returned as a single tensor as target.
        :param species_subset: The species to include in the dataset as targets. If None, all species will be included.
        :param dataset_type: The type of dataset to use. If you want to measure the MIC prediction, use 'MIC'. If you want to accuracy at predicting whether a peptide is hemolytic or cytotoxic, use 'Hemolytic' or 'Cytotoxic' respectively.
        :param show_all: If true, even if the modified_termini, unusual amino acids or D-amino acids are not used, the dataset will still return the sequences. Otherwise, it will skip those sequences.
        """
        super().__init__()
        self.raw_dataset = self._load_dataset("../data/build/dataset.json")

    @property
    def inputs(self):
        pass

    @property
    def targets(self):
        pass

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        pass

    def get_train_mask(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Returns a mask indicating which sequences can be in the training set because they are not too similar to any
        other sequence in the test set.
        :param sequences: The sequences to check.
        :return: True if the sequence is allowed and False otherwise.
        """
        pass

    @staticmethod
    def _load_dataset(path: str):
        with open(path, "r") as f:
            data = json.load(f)

        dataset = []
        for sample in data:
            sequence = sample['sequence']
            n_terminus = sample['n_terminus']
            c_terminus = sample['c_terminus']
            unusual_aa = sample['unusual_aa']
            unusual_aa_names = sample['unusual_aa_names']
            targets = sample['targets']
            hemolytic = sample['hemolytic']
            cytotoxic = sample['cytotoxic']

            dataset.append(Sample(
                id_=sample['id'],
                sequence=sequence,
                n_terminus=n_terminus,
                c_terminus=c_terminus,
                unusual_aa=unusual_aa,
                unusual_aa_names=unusual_aa_names,
                targets=targets,
                hemolytic=hemolytic,
                cytotoxic=cytotoxic
            ))
        return dataset