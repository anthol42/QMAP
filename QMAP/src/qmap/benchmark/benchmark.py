import numpy as np
from torch.utils.data import Dataset
from typing import List, Literal, Sequence, Optional, Tuple
from .sample import Sample
import json

from ..toolkit.aligner import Encoder, align_db

class QMAPBenchmark(Dataset):
    """
    Class representing the QMAP benchmark testing dataset. It is a subclass of `torch.utils.data.Dataset`, so it can be
    easily used with PyTorch's DataLoader. However, it is easy to extract the sequences and the labels from it to use
    it with other libraries such as tensorflow or keras.
    """
    def __init__(self,
                 split: int = 0,
                 threshold: Literal[55, 60] = 60,
                 *,
                 modified_termini: bool = False,
                 unusual_aa: bool = False,
                 d_amino_acids: bool = False,
                 specie_as_input: bool = False,
                 species_subset: Optional[List[str]] = None,
                 dataset_type: Literal['MIC', 'Hemolytic', 'Cytotoxic'] = 'MIC',
                 show_all: bool = False,
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
        self.split = split
        self.threshold = threshold / 100  # Convert to fraction

        self.raw_dataset = self._load_dataset(f"../data/build/benchmark_threshold-{threshold}_split-{split}.json")
        self.modified_termini = modified_termini
        self.unusual_aa = unusual_aa
        self.d_amino_acids = d_amino_acids
        self.specie_as_input = specie_as_input
        self.species_subset = species_subset if species_subset is not None else []
        self.dataset_type = dataset_type
        self.show_all = show_all

        if self.dataset_type not in ['MIC', 'Hemolytic', 'Cytotoxic']:
            raise ValueError("dataset_type must be one of 'MIC', 'Hemolytic', or 'Cytotoxic'.")

        # Initialize data containers
        self.sequences, self.species, self.targets = [], [], []
        self.c_termini, self.n_termini, self.unusual_aa = [], [], []
        self.max_targets, self.min_targets = [], []


        # Process samples
        for sample in self.raw_dataset:
            if self.dataset_type == 'MIC':
                if self.specie_as_input:
                    # Add one entry per species
                    for specie, (consensus, minMIC, maxMIC) in sample.targets:
                        if specie in self.species_subset:
                            self._add_sample_data(sample, specie, consensus, minMIC, maxMIC)
                else:
                    # Create multi-target array for all species
                    label = {target: np.nan for target in self.species_subset}
                    label_min = {target: np.nan for target in self.species_subset}
                    label_max = {target: np.nan for target in self.species_subset}

                    for target, (value, minMIC, maxMIC) in sample.targets.items():
                        if target in self.species_subset:
                            label[target] = value
                            label_min[target] = minMIC
                            label_max[target] = maxMIC

                    label_array = np.array(list(label.values()))
                    if not np.isnan(label_array).all():
                        self._add_sample_data(sample, None, label_array,
                                        np.array(list(label_min.values())),
                                        np.array(list(label_max.values())))
            else:
                # Handle Hemolytic/Cytotoxic
                target_value = getattr(sample, self.dataset_type.lower(), None)
                if target_value is not None:
                    self._add_sample_data(sample, self.dataset_type if self.specie_as_input else None, target_value)

        # Set attributes
        self.species = self.species if self.specie_as_input else None
        self.max_targets = self.max_targets if self.dataset_type == 'MIC' else None
        self.min_targets = self.min_targets if self.dataset_type == 'MIC' else None

        # Remove samples that have modification that are not specified
        if not self.show_all:
            self._filter_dataset()

    @property
    def inputs(self) -> Tuple[List[str], ...]:
        """
        Depending on the initialization parameters, this property will return different values. It will return:
        - The sequence (str) always
        - The specie (str) if specie_as_input is True
        - The N terminus (str) if modified_termini is True
        - The C terminus (str) if modified_termini is True
        - The unusual amino acids (dict) if unusual_aa is True
        :return: All the inputs for the model.
        """
        out = [self.sequences]
        if self.specie_as_input:
            out.append(self.species)
        if self.modified_termini:
            out.append(self.n_termini)
            out.append(self.c_termini)
        if self.unusual_aa:
            out.append(self.unusual_aa)
        return tuple(out)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx) -> Tuple:
        """
        Depending on the initialization parameters, this method will return different values. It will return:
        - The sequence (str) always
        - The specie (str) if specie_as_input is True
        - The N terminus (str) if modified_termini is True
        - The C terminus (str) if modified_termini is True
        - The unusual amino acids (dict) if unusual_aa is True
        - The target (float or np.ndarray) always

        :param idx: The index of the sample to retrieve.
        :return: The inputs for a single sample and the target.
        """
        sequence = self.sequences[idx]
        out = [sequence]
        if self.specie_as_input:
            out.append(self.species[idx])
        if self.modified_termini:
            out.append(self.n_termini[idx])
            out.append(self.c_termini[idx])
        if self.unusual_aa:
            out.append(self.unusual_aa[idx])

        target = self.targets[idx]
        return *out, target

    def get_train_mask(self, sequences: List[str],
                       encoder_batch_size: int = 512,
                       align_batch_size: int = 0,
                       force_cpu: bool = False
                       ) -> np.ndarray:
        """
        Returns a mask indicating which sequences can be in the training set because they are not too similar to any
        other sequence in the test set.
        :param sequences: The sequences to check.
        :param encoder_batch_size: The batch size to use for encoding. Change this value if you run out of memory.
        :param align_batch_size: The batch size to use for alignment. If 0, the batch size will be the full dataset. Change this
        value if you run out of memory.
        :param force_cpu: If True, the alignment will be forced to run on CPU.
        :return: True if the sequence is allowed and False otherwise.
        """
        # Step 1: Encode the sequences
        encoder = Encoder(force_cpu=force_cpu)
        bench_db = encoder.encode(self.sequences)
        ds_db = encoder.encode(sequences, batch_size=encoder_batch_size)

        # Step 2: Align the databases
        alignments = align_db(ds_db, bench_db, batch=align_batch_size, device='cpu')
        mask = alignments.alignment_matrix.max(axis=1) > self.threshold

        return ~mask

    def accuracy(self, predictions: np.ndarray) -> float:
        """
        Compute the accuracy of the predictions. A good prediction is one that is within the MIC range. This method
        only work with MIC datasets.
        :param predictions: The predictions to evaluate. It should have the same length and order as this dataset.
        :return: The accuracy of the predictions.
        """
        good = np.logical_and(predictions >= self.min_targets, predictions <= self.max_targets)
        return np.sum(good) / len(good)

    def _filter_dataset(self):
        mask = [True] * len(self.sequences)
        for i in range(len(self.sequences)):
            if not self.modified_termini and (self.n_termini[i] is not None or self.c_termini[i] is not None):
                mask[i] = False
            if not self.unusual_aa and self.unusual_aa[i]:
                mask[i] = False
            if not self.d_amino_acids and any(aa.islower() for aa in self.sequences[i]):
                mask[i] = False

        self.sequences = [seq for i, seq in enumerate(self.sequences) if mask[i]]
        self.species = [spec for i, spec in enumerate(self.species) if mask[i]] if self.species is not None else None
        self.targets = [tgt for i, tgt in enumerate(self.targets) if mask[i]]
        self.c_termini = [ct for i, ct in enumerate(self.c_termini) if mask[i]]
        self.n_termini = [nt for i, nt in enumerate(self.n_termini) if mask[i]]
        self.unusual_aa = [ua for i, ua in enumerate(self.unusual_aa) if mask[i]]
        self.max_targets = [mt for i, mt in enumerate(self.max_targets) if mask[i]] if self.max_targets is not None else None
        self.min_targets = [mt for i, mt in enumerate(self.min_targets) if mask[i]] if self.min_targets is not None else None

    def _add_sample_data(self, sample, specie=None, target=None, min_mic=None, max_mic=None):
        """Helper to add sample data to containers."""
        self.sequences.append(sample.sequence)
        self.species.append(specie or self.dataset_type)
        self.targets.append(target)
        self.c_termini.append(sample.c_terminus)
        self.n_termini.append(sample.n_terminus)
        self.unusual_aa.append(sample.unusual_aa)
        if min_mic is not None:
            self.min_targets.append(min_mic)
            self.max_targets.append(max_mic)

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