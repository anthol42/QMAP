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
        self.modified_termini = modified_termini
        self.unusual_aa = unusual_aa
        self.d_amino_acids = d_amino_acids
        self.specie_as_input = specie_as_input
        self.species_subset = species_subset if species_subset is not None else []
        self.dataset_type = dataset_type
        self.show_all = show_all

        if self.dataset_type not in ['MIC', 'Hemolytic', 'Cytotoxic']:
            raise ValueError("dataset_type must be one of 'MIC', 'Hemolytic', or 'Cytotoxic'.")

        if self.specie_as_input:
            # Remake the sequences
            sequences = []
            species = []
            targets = []
            c_termini = []
            n_termini = []
            unusual_aa = []
            if self.dataset_type == 'MIC':
                for sample in self.raw_dataset:
                    for specie, value in sample.targets:
                        if specie in species_subset:
                            sequences.append(sample.sequence)
                            species.append(specie)
                            targets.append(value[0])
                            c_termini.append(sample.c_terminus)
                            n_termini.append(sample.n_terminus)
                            unusual_aa.append(sample.unusual_aa)
            else:
                for sample in self.raw_dataset:
                    if self.dataset_type == 'Hemolytic':
                        if sample.hemolytic is not None:
                            targets.append(sample.hemolytic)
                            species.append(self.dataset_type)
                            sequences.append(sample.sequence)
                            c_termini.append(sample.c_terminus)
                            n_termini.append(sample.n_terminus)
                            unusual_aa.append(sample.unusual_aa)
                    elif self.dataset_type == 'Cytotoxic':
                        if sample.cytotoxic is not None:
                            targets.append(sample.cytotoxic)
                            species.append(self.dataset_type)
                            sequences.append(sample.sequence)
                            c_termini.append(sample.c_terminus)
                            n_termini.append(sample.n_terminus)
                            unusual_aa.append(sample.unusual_aa)
            self.sequences = sequences
            self.species = species
            self.targets = targets
            self.c_termini = c_termini
            self.n_termini = n_termini
            self.unusual_aa = unusual_aa

        else:
            sequences = []
            species = None
            targets = []
            c_termini = []
            n_termini = []
            unusual_aa = []
            for sample in self.raw_dataset:
                label = {target: np.nan for target in species_subset}
                for target, (value, minMIC, maxMIC) in sample.targets.items():
                    if target in species_subset:
                        label[target] = value
                label_array = np.array([value for value in label.values()])
                if not np.isnan(label_array).all():
                    sequences.append(sample.sequence)
                    targets.append(label_array)
                    c_termini.append(sample.c_terminus)
                    n_termini.append(sample.n_terminus)
                    unusual_aa.append(sample.unusual_aa)

            self.sequences = sequences
            self.species = species
            self.targets = targets
            self.c_termini = c_termini
            self.n_termini = n_termini
            self.unusual_aa = unusual_aa




    @property
    def inputs(self):
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