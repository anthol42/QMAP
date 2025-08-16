import numpy as np
from typing import List, Literal, Sequence, Optional, Tuple
from .sample import Sample
import json
from huggingface_hub import hf_hub_download

from .subset import BenchmarkSubset
from ..toolkit.utils import sequence_entropy
from ..toolkit.aligner import Encoder, align_db

COMPLEXITY_THRESHOLD = 2.62 # The median of the DBAASP dataset.

class QMAPBenchmark(BenchmarkSubset):
    """
    Class representing the QMAP benchmark testing dataset. It is a subclass of `torch.utils.data.Dataset`, so it can be
    easily used with PyTorch's DataLoader. However, it is easy to extract the sequences and the labels from it to use
    it with other libraries such as tensorflow or keras. You can do this by using the `inputs` and `targets` attributes.

    Once you have the predictions of your model, you can call the `compute_metrics` method to compute the metrics of
    your model on the given test set. You can also evaluate the performances of your model on subsets like
    `high_complexity`, `low_complexity` or `high_efficiency`. You can select the subset from the attributes of the
    same name. The subset have the same interface as the `QMAPBenchmark` class, so you can use them with the same
    methods!

    To use the benchmark, you must select at least the split (from 0 to 4) and the threshold (55 or 60). It is highly
    recommended to test your model on all splits to get a better estimate of its real-world performance and to
    accurately compare it with other models. To do so, you must use the same hyperparameters, but change the training
    and validation dataset. For each split, use the `get_train_mask` method to get a mask indicating which sequences
    can be used in the training set and validation set. This mask will be True for sequences that are allowed in the
    training set and False for sequences that are too similar to a sequence in the test set. Train your model on the
    subset of your training dataset where the mask is True and evaluate it on the benchmark dataset. Do this for all
    splits. See the example section for more details.

    Thresholds:

    - 55: This threshold enables a split that is considered natural as it have a maximum identity distribution between
    the train and test set similar to natural independent peptide datasets.

    - 60: This threshold enables a harder split because it increases the diversity of the test set. Even if the
    maximum identity distribution is shifted to more similar sequences between train and test compared to the natural
    split (55), it is considered conservative as models do not perform as well on this split. It is recommended to use
    this split as it gives a more conservative estimate of the model's real-world performance.
    """
    def __init__(self,
                 split: int = 0,
                 threshold: Literal[55, 60] = 60,
                 *,
                 modified_termini: bool = False,
                 unusual_aa: bool = False,
                 d_amino_acids: bool = False,
                 forbidden_aa: Optional[Sequence[str]] = tuple(),
                 specie_as_input: bool = False,
                 species_subset: Optional[List[str]] = None,
                 dataset_type: Literal['MIC', 'Hemolytic', 'Cytotoxic'] = 'MIC',
                 show_all: bool = False,
                 ):
        """
        :param modified_termini: If True, the dataset will return the N and C terminus smiles string. Otherwise, sequences containing modified termini will be skipped.
        :param unusual_aa: If True, the dataset will return the unusual amino acids as a dictionary of positions and their. Otherwise, sequences containing unusual amino acids will be skipped.
        :param d_amino_acids: If True, the dataset will return sequences with D-amino acids. Otherwise, sequences containing D-amino acids will be skipped.
        :param forbidden_aa: A sequence of amino acids that should not be passed to the model. If present in the sequence, the sequence will be dropped.
        :param specie_as_input: If True, the dataset will return a tuple of (sequence, specie) and the target will be a scalar. Otherwise, all species in species_subset will be returned as a single tensor as target.
        :param species_subset: The species to include in the dataset as targets. If None, all species will be included.
        :param dataset_type: The type of dataset to use. If you want to measure the MIC prediction, use 'MIC'. If you want to accuracy at predicting whether a peptide is hemolytic or cytotoxic, use 'Hemolytic' or 'Cytotoxic' respectively.
        :param show_all: If true, even if the modified_termini, unusual amino acids or D-amino acids are not used, the dataset will still return the sequences. Otherwise, it will skip those sequences.
        """
        if split not in [0, 1, 2, 3, 4]:
            raise ValueError("split must be one of 0, 1, 2, 3, or 4.")
        if threshold not in [55, 60]:
            raise ValueError("threshold must be either 55 or 60.")
        self.split = split
        self.threshold = threshold / 100  # Convert to fraction

        path = hf_hub_download(
            repo_id="anthol42/qmap_benchmark_2025",
            filename=f"benchmark_threshold-{threshold}_split-{split}.json",
            repo_type="dataset"
        )
        self.raw_dataset = self._load_dataset(path)
        self.modified_termini = modified_termini
        self.allow_unusual_aa = unusual_aa
        self.d_amino_acids = d_amino_acids
        self.forbidden_aa = forbidden_aa
        self.specie_as_input = specie_as_input
        self.species_subset = species_subset if species_subset is not None else []
        self.dataset_type = dataset_type
        self.show_all = show_all

        if self.dataset_type not in ['MIC', 'Hemolytic', 'Cytotoxic']:
            raise ValueError("dataset_type must be one of 'MIC', 'Hemolytic', or 'Cytotoxic'.")

        # Initialize data containers
        self.sequences, self.species, self._targets = [], [], []
        self.c_termini, self.n_termini, self.unusual_aa = [], [], []
        self.max_targets, self.min_targets = [], []


        # Process samples
        for sample in self.raw_dataset:
            if self.dataset_type == 'MIC':
                if self.specie_as_input:
                    # Add one entry per species
                    for specie, (consensus, minMIC, maxMIC) in sample.targets.items():
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
                target_value = getattr(sample, self.dataset_type.lower(), np.nan)
                if not np.isnan(target_value):
                    self._add_sample_data(sample, self.dataset_type if self.specie_as_input else None, target_value)

        # Set attributes
        self.species = self.species if self.specie_as_input else None
        self.max_targets = self.max_targets if self.dataset_type == 'MIC' else None
        self.min_targets = self.min_targets if self.dataset_type == 'MIC' else None

        super().__init__(self.split, threshold, self.dataset_type,
                         self.sequences, self.species, self._targets, self.c_termini, self.n_termini, self.unusual_aa,
                         self.max_targets, self.min_targets,
                         modified_termini=modified_termini,
                         allow_unusual_aa=unusual_aa,
                         specie_as_input=specie_as_input)

        # Remove samples that have modification that are not specified
        if not self.show_all:
            self._filter_dataset()


    @property
    def low_complexity(self) -> BenchmarkSubset:
        """
        Returns a subset of the benchmark with only low complexity sequences. A low complexity sequence is defined as a
        sequence that have an entropy below the median complexity of the DBAASP dataset.
        """
        complexity = np.array([sequence_entropy(seq) for seq in self.sequences])

        mask = complexity < COMPLEXITY_THRESHOLD

        return BenchmarkSubset(
            split=self.split,
            threshold=int(100*self.threshold),
            dataset_type=self.dataset_type,
            sequences=[seq for i, seq in enumerate(self.sequences) if mask[i]],
            species=[spec for i, spec in enumerate(self.species) if mask[i]] if self.species is not None else None,
            targets=[tgt for i, tgt in enumerate(self._targets) if mask[i]],
            c_termini=[ct for i, ct in enumerate(self.c_termini) if mask[i]],
            n_termini=[nt for i, nt in enumerate(self.n_termini) if mask[i]],
            unusual_aa=[ua for i, ua in enumerate(self.unusual_aa) if mask[i]],
            max_targets=[mt for i, mt in enumerate(self.max_targets) if mask[i]] if self.max_targets is not None else None,
            min_targets=[mt for i, mt in enumerate(self.min_targets) if mask[i]] if self.min_targets is not None else None,

            modified_termini=self.modified_termini,
            allow_unusual_aa=self.allow_unusual_aa,
            specie_as_input=self.specie_as_input,
        )

    @property
    def high_complexity(self) -> BenchmarkSubset:
        """
        Return a subset of the benchmark with only high complexity sequences. A high complexity sequence is defined as a
        sequence that have an entropy superior to the median complexity of the DBAASP dataset (Similar to natural peptides).
        """
        complexity = np.array([sequence_entropy(seq) for seq in self.sequences])

        mask = complexity >= COMPLEXITY_THRESHOLD

        return BenchmarkSubset(
            split=self.split,
            threshold=int(100 * self.threshold),
            dataset_type=self.dataset_type,
            sequences=[seq for i, seq in enumerate(self.sequences) if mask[i]],
            species=[spec for i, spec in enumerate(self.species) if mask[i]] if self.species is not None else None,
            targets=[tgt for i, tgt in enumerate(self._targets) if mask[i]],
            c_termini=[ct for i, ct in enumerate(self.c_termini) if mask[i]],
            n_termini=[nt for i, nt in enumerate(self.n_termini) if mask[i]],
            unusual_aa=[ua for i, ua in enumerate(self.unusual_aa) if mask[i]],
            max_targets=[mt for i, mt in enumerate(self.max_targets) if
                         mask[i]] if self.max_targets is not None else None,
            min_targets=[mt for i, mt in enumerate(self.min_targets) if
                         mask[i]] if self.min_targets is not None else None,

            modified_termini=self.modified_termini,
            allow_unusual_aa=self.allow_unusual_aa,
            specie_as_input=self.specie_as_input,
        )

    @property
    def high_efficiency(self) -> BenchmarkSubset:
        """
        Return a subset of the benchmark with only high efficiency sequences. A high efficiency sequence is defined as
        a sample that have a MIC under 10µM.
        """
        if self.dataset_type != "MIC":
            raise AttributeError("high_efficiency attribute is only available for MIC datasets")

        mask = self.targets < 10

        return BenchmarkSubset(
            split=self.split,
            threshold=int(100 * self.threshold),
            dataset_type=self.dataset_type,
            sequences=[seq for i, seq in enumerate(self.sequences) if mask[i]],
            species=[spec for i, spec in enumerate(self.species) if mask[i]] if self.species is not None else None,
            targets=[tgt for i, tgt in enumerate(self._targets) if mask[i]],
            c_termini=[ct for i, ct in enumerate(self.c_termini) if mask[i]],
            n_termini=[nt for i, nt in enumerate(self.n_termini) if mask[i]],
            unusual_aa=[ua for i, ua in enumerate(self.unusual_aa) if mask[i]],
            max_targets=[mt for i, mt in enumerate(self.max_targets) if
                         mask[i]] if self.max_targets is not None else None,
            min_targets=[mt for i, mt in enumerate(self.min_targets) if
                         mask[i]] if self.min_targets is not None else None,

            modified_termini=self.modified_termini,
            allow_unusual_aa=self.allow_unusual_aa,
            specie_as_input=self.specie_as_input,
        )

    def get_train_mask(self, sequences: List[str],
                       encoder_batch_size: int = 512,
                       align_batch_size: int = 0,
                       force_cpu: bool = False
                       ) -> np.ndarray:
        """
        Returns a mask indicating which sequences can be in the training set because they are not too similar to any
        other sequence in the test set. It returns a boolean mask where True means that the sequence is allowed in the
        training / validation set and False means that the sequence is too similar to a sequence in the test set and
        must be excluded.
        :param sequences: The sequences to check.
        :param encoder_batch_size: The batch size to use for encoding. Change this value if you run out of memory.
        :param align_batch_size: The batch size to use for alignment. If 0, the batch size will be the full dataset. Change this value if you run out of memory.
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

    def _filter_dataset(self):
        mask = [True] * len(self.sequences)
        for i in range(len(self.sequences)):
            if not self.modified_termini and (self.n_termini[i] is not None or self.c_termini[i] is not None):
                mask[i] = False
            if not self.allow_unusual_aa and len(self.unusual_aa[i]) > 0:
                mask[i] = False
            if not self.d_amino_acids and any(aa.islower() for aa in self.sequences[i]):
                mask[i] = False
            if any(aa in self.forbidden_aa for aa in self.sequences[i]):
                mask[i] = False

        self.sequences = [seq for i, seq in enumerate(self.sequences) if mask[i]]
        self.species = [spec for i, spec in enumerate(self.species) if mask[i]] if self.species is not None else None
        self._targets = [tgt for i, tgt in enumerate(self._targets) if mask[i]]
        self.c_termini = [ct for i, ct in enumerate(self.c_termini) if mask[i]]
        self.n_termini = [nt for i, nt in enumerate(self.n_termini) if mask[i]]
        self.unusual_aa = [ua for i, ua in enumerate(self.unusual_aa) if mask[i]]
        self.max_targets = [mt for i, mt in enumerate(self.max_targets) if mask[i]] if self.max_targets is not None else None
        self.min_targets = [mt for i, mt in enumerate(self.min_targets) if mask[i]] if self.min_targets is not None else None

    def _add_sample_data(self, sample, specie=None, target=None, min_mic=None, max_mic=None):
        """Helper to add sample data to containers."""
        self.sequences.append(sample.sequence)
        self.species.append(specie or self.dataset_type)
        self._targets.append(target)
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
            sequence = sample['Sequence']
            n_terminus = sample['N Terminus']
            n_terminus_name = sample['N Terminus Name']
            c_terminus = sample['C Terminus']
            c_terminus_name = sample['C Terminus Name']
            unusual_aa = sample['Unusual Amino Acids']
            unusual_aa_names = sample['Unusual Amino Acids Names']
            targets = sample['Targets']
            hemolytic = sample['Hemolitic Activity']
            cytotoxic = sample['Cytotoxic Activity']

            dataset.append(Sample(
                id_=sample['ID'],
                sequence=sequence,
                n_terminus=n_terminus,
                n_terminus_name=n_terminus_name,
                c_terminus=c_terminus,
                c_terminus_name=c_terminus_name,
                unusual_aa=unusual_aa,
                unusual_aa_names=unusual_aa_names,
                targets=targets,
                hemolytic=hemolytic,
                cytotoxic=cytotoxic
            ))
        return dataset