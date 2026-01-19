import pandas as pd
import numpy as np
from typing import Callable, Optional, Literal, Union
from .sample import Sample
from .bond import Bond
from .target import Target
from .hemolytic import HemolyticActivity
from .filters import (filter_bacteria, filter_efficiency_below, filter_hc50, filter_canonical_only,
                      filter_common_only, filter_l_aa_only, filter_terminal_modification, filter_bond)

def _format_sample_line(sample: Sample) -> str:
    free_str = 'Free'
    return f'{str(sample.id).ljust(10, " ")} {sample.sequence[:40].ljust(40, " ")} {(sample.nterminal or free_str).ljust(12)} {(sample.cterminal or free_str).ljust(12)} {str(len(sample.targets)).ljust(10)} {sample.hc50.consensus if sample.hc50 else np.nan:.2f}'


class DBAASPDataset:
    """
    Class representing a the DBAASP dataset used to build the benchmark.

    ## Example:
    dataset = DBAASPDataset(...)

    dataset_filtered = (dataset
               .with_bacterial_targets(["Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"])
               .with_efficiency_below(10.0)
               .with_common_only()
               .with_l_aa_only()
               .with_terminal_modification(None, None)
               )
    print(dataset)

    # You can also extend the dataset with your own samples:
    dataset_extended = dataset_filtered + DBAASPDataset(your_own_data)
    # Or with the extend method
    dataset_extended = dataset_filtered.extend(DBAASPDataset(your_own_data))

    # You can also index the dataset like a list, or a numpy array:
    first_sample = dataset[0]
    some_samples = dataset[1:10]
    boolean_indexed_samples = dataset[np.array([True, False, True, False, ...])]
    only_certain_samples = dataset[[0, 2, 5, 7]]
    """
    def __init__(self, data: list[dict]):
        """
        When constructing the dataset from raw data, the data should be a list of dictionaries with the following format:
        {
            'id': int,
            'sequence': str,
            'smiles': list[str],
            'nterminal': Optional['ACT'],
            'cterminal': Optional['AMD'],
            'bonds': list[tuple[int, int, 'DSB' | 'AMD']],
            'targets': dict[str, tuple[float, float, float]],
            'hemolytic_hc50': Optional[tuple[float, float, float]]
        }
        """
        self.samples = [Sample.FromDict(sample_data) for sample_data in data]

    @property
    def sequences(self) -> list[str]:
        """
        Return the list of sequences in the dataset.
        """
        return [sample.sequence for sample in self.samples]

    @property
    def ids(self) -> list[int]:
        """
        Return the list of DBAASP IDs in the dataset.
        """
        return [sample.id for sample in self.samples]

    @property
    def smiles(self) -> list[list[str]]:
        """
        Return the list of SMILES strings in the dataset.
        """
        return [sample.smiles for sample in self.samples]

    @property
    def nterminals(self) -> list[Optional[str]]:
        """
        Return the list of N-terminal modifications in the dataset.
        """
        return [sample.nterminal for sample in self.samples]

    @property
    def cterminals(self) -> list[Optional[str]]:
        """
        Return the list of C-terminal modifications in the dataset.
        """
        return [sample.cterminal for sample in self.samples]

    @property
    def bonds(self) -> list[Bond]:
        """
        Return the list of bonds in the dataset.
        """
        return [sample.bonds for sample in self.samples]

    @property
    def targets(self) -> list[list[Target]]:
        """
        Return the list of targets in the dataset.
        """
        return [list(sample.targets.values) for sample in self.samples]

    @property
    def hc50s(self) -> list[Optional[HemolyticActivity]]:
        """
        Return the list of hemolytic activities (HC50) in the dataset.
        """
        return [sample.hc50 for sample in self.samples]

    @classmethod
    def FromSamples(cls, samples: list[Sample]) -> "DBAASPDataset":
        """
        Load a dataset from a list of Sample objects, instead of from raw data.
        """
        self = DBAASPDataset([])
        self.samples = samples
        return self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: Union[int, slice, np.ndarray, list[bool], list[int]]):
        if isinstance(idx, np.ndarray):
            if idx.dtype == bool:
                return DBAASPDataset.FromSamples([sample for sample, keep in zip(self.samples, idx) if keep])
            else:
                return DBAASPDataset.FromSamples([self.samples[i] for i in idx.tolist()])
        elif isinstance(idx, list) and all(isinstance(i, bool) for i in idx):
            return DBAASPDataset.FromSamples([sample for sample, keep in zip(self.samples, idx) if keep])
        elif isinstance(idx, list) and all(isinstance(i, int) for i in idx):
            return DBAASPDataset.FromSamples([self.samples[i] for i in idx])
        elif isinstance(idx, int):
            return self.samples[idx]
        elif isinstance(idx, slice):
            return self.samples[idx]
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self):
        return f"DBAASPDataset({self.samples})"

    def __str__(self):
        columns = f'{"DBAASP ID".ljust(10, " ")} {"Sequence".ljust(40, " ")} {"N Terminal".ljust(12)} {"C Terminal".ljust(12)} {"# Targets".ljust(10, " ")} {"HC50".ljust(6, " ")}'
        if len(self) > 10:
            first_four = [_format_sample_line(sample) for sample in self.samples[:4]]
            middle = len(columns) // 2 * " " + "..."
            last_four = [_format_sample_line(sample) for sample in self.samples[-4:]]
            return columns + "\n"+ "\n".join(first_four) + "\n" + middle + "\n" + "\n".join(last_four) + f"\n\nTotal of {len(self)} samples"
        else:
            lines = [_format_sample_line(sample) for sample in self.samples]
            return columns + "\n" + "\n".join(lines) + f"\n\nTotal of {len(self)} samples"

    def __add__(self, other):
        return self.extend(other)

    def extend(self, other: 'DBAASPDataset') -> 'DBAASPDataset':
        """
        Extend the dataset with another DBAASPDataset.
        """
        extended_samples = self.samples + other.samples
        return DBAASPDataset.FromSamples(extended_samples)

    def tabular(self, columns: list[str]) -> pd.DataFrame:
        """
        Convert the sample to tabular data. Only these fields are supported:
        - id: DBAASP ID
        - sequence: noncanonical: O is Ornithine, B is DAB
        - smiles: Note that only the first SMILES string is used
        - nterminal: None or ACT
        - cterminal: None or AMD
        - targets <target_name>: Note that only the consensus value is used
        - hc50: Note that only the consensus value is used

        ## Example:
        ```
        df = dataset.tabular(["id", "sequence", "nterminal", "cterminal", "hc50", "Escherichia coli"])
        print(df)
        ```
        """
        rows = [sample.tabular(columns) for sample in self.samples]
        return pd.DataFrame(rows, columns=columns)


    def filter(self, mapper: Callable[[Sample], bool]) -> 'DBAASPDataset':
        """
        Filter the dataset using a mapper function that takes a Sample and returns a boolean.
        A sample is kept if the mapper returns True, and drop otherwise.
        """
        filtered_samples = [sample for sample in self.samples if mapper(sample)]
        return DBAASPDataset.FromSamples(filtered_samples)

    def with_bacterial_targets(self, allowed: list[str]) -> 'DBAASPDataset':
        """
        Keep only samples that have at least one bacterial target in the allowed list.
        """
        return self.filter(filter_bacteria(allowed))

    def with_efficiency_below(self, threshold: float) -> 'DBAASPDataset':
        """
        Keep only samples that have at least one bacterial target with efficiency below the given threshold (in µM).
        """
        return self.filter(filter_efficiency_below(threshold))

    def with_hc50(self) -> 'DBAASPDataset':
        """
        Keep only samples that have hemolytic activity (HC50) reported.
        """
        return self.filter(filter_hc50())

    def with_canonical_only(self) -> 'DBAASPDataset':
        """
        Keep only samples that have only canonical amino acids in their sequence.
        Sequences with non-canonical amino acids represented by the letters O (Ornithine) and B (DAB) or X are
        filtered out.
        """
        return self.filter(filter_canonical_only())

    def with_common_only(self) -> 'DBAASPDataset':
        """
        Keep only samples that have only common amino acids in their sequence: canonical amino acids and
        Ornithin (O) and DAB (B).
        """
        return self.filter(filter_common_only())

    def with_l_aa_only(self) -> 'DBAASPDataset':
        """
        Keep only samples that have only L-amino acids in their sequence.
        D-amino acids are represented by lowercase letters.
        """
        return self.filter(filter_l_aa_only())

    def with_terminal_modification(self, nterminal: Optional[bool], cterminal: Optional[bool]) -> 'DBAASPDataset':
        """
        Keep only samples that have the specified terminal modifications.
        :param nterminal: N-terminal modification to filter by. Use None for free N-terminus, or 'ACT' for acetylation. Use True to keep only sequence having 'ACT' modification, or False to keep only sequences with free N-terminus.
        :param cterminal: C-terminal modification to filter by. Use None for free C-terminus, or 'AMD' for amidation. Use True to keep only sequence having 'AMD' modification, or False to keep only sequences with free C-terminus.
        """
        return self.filter(filter_terminal_modification(nterminal, cterminal))

    def with_bond(self, bond_type: list[Literal['DSB', 'AMD', None]]) -> 'DBAASPDataset':
        """
        Keep only samples that have bonds within the specified bond types. None means no bond is allowed.
        For example, to keep only samples with disulfide bonds, use bond_type=['DSB']. To keep only samples with no
        bonds, use bond_type=[None].
        """
        return self.filter(filter_bond(bond_type))