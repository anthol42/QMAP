from .sample import Sample
from typing import Optional, Literal
import math

def filter_bacteria(allowed: list[str]):
    """
    Keep only the samples that have a target within the allowed list of bacterial species.

    If at least one target specie of the sample is in the allowed list, the sample is kept, otherwise it is
    filtered out.
    """
    def filter_fn(sample: Sample) -> bool:
        for target in sample.targets:
            if target in allowed:
                return True
        return False
    return filter_fn

def filter_efficiency_below(threshold: Optional[float]):
    """
    Keep only the samples that have at least one target with a consensus MIC value below the given threshold.
    """
    def filter_fn(sample: Sample) -> bool:
        for target in sample.targets.values():
            if target.consensus < threshold:
                return True
        return False
    return filter_fn

def filter_hc50():
    """
    Keep only the samples that have a hemolytic activity (HC50)
    """
    def filter_fn(sample: Sample) -> bool:
        return not math.isnan(sample.hc50.consensus)
    return filter_fn

def filter_canonical_only():
    """
    Keep only the samples that have no non-canonical amino acids in their sequence.
    Non-canonical amino acids are represented by the letters O (Ornithine) and B (DAB).
    """
    def filter_fn(sample: Sample) -> bool:
        return 'O' not in sample.sequence.upper() and 'B' not in sample.sequence.upper() and 'X' not in sample.sequence.upper()
    return filter_fn

def filter_common_only():
    """
    Keep only the samples that have only common amino acids in their sequence: canonical amino acids and Ornithin (O) and DAB (B).
    """
    def filter_fn(sample: Sample) -> bool:
        return 'X' not in sample.sequence.upper()
    return filter_fn


def filter_l_aa_only():
    """
    Keep only the samples that have only L-amino acids in their sequence.
    D-amino acids are represented by lowercase letters.
    """
    def filter_fn(sample: Sample) -> bool:
        return all(c.isupper() for c in sample.sequence)
    return filter_fn

def filter_terminal_modification(n_terminal: Optional[bool], c_terminal: Optional[bool]):
    """
    Filter samples that do not have the specified terminal modifications.
    :param n_terminal: If True, keep only samples with N-terminal modification (ACT).
                       If False, keep only samples without N-terminal modification.
                       If None, do not filter based on N-terminal modification.
    :param c_terminal: If True, keep only samples with C-terminal modification (AMD).
                          If False, keep only samples without C-terminal modification.
                          If None, do not filter based on C-terminal modification.
    """
    def filter_fn(sample: Sample) -> bool:
        if n_terminal is not None:
            has_n_term = sample.nterminal is not None
            if n_terminal != has_n_term:
                return False
        if c_terminal is not None:
            has_c_term = sample.cterminal is not None
            if c_terminal != has_c_term:
                return False
        return True
    return filter_fn

def filter_min_length(min_length: int):
    """
    Keep only the samples that have a sequence length greater than or equal to the given minimum length.
    """
    def filter_fn(sample: Sample) -> bool:
        return len(sample.sequence) >= min_length
    return filter_fn

def filter_max_length(max_length: int):
    """
    Keep only the samples that have a sequence length less than or equal to the given maximum length.
    """
    def filter_fn(sample: Sample) -> bool:
        return len(sample.sequence) <= max_length
    return filter_fn

def filter_bond(allowed_bond_types: list[Literal['DSB', 'AMD', None]]):
    """
    Filter out samples that have bonds that are not in the allowed bond types. None means no intrachain bond is allowed.

    For example, to filter out samples that have an intrachain bond, we would do:
    ```
    dataset = dataset.filter(filter_bond([None]))
    ```
    """
    def filter_fn(sample: Sample) -> bool:
        if len(sample.bonds) == 0 and None not in allowed_bond_types:
            return False

        for bond in sample.bonds:
            if bond.bond_type not in allowed_bond_types:
                return False
        return True
    return filter_fn