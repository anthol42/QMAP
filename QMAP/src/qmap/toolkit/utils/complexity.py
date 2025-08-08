from collections import Counter
from math import log2

def sequence_entropy(sequence: str) -> float:
    """
    Calculate the Shannon entropy of a sequence.
    :param sequence: The sequence to calculate the entropy for
    :return: The Shannon entropy of the sequence
    """

    counts = Counter(sequence)
    total = len(sequence)
    probabilities = [count / total for count in counts.values()]
    return -sum(p * log2(p) for p in probabilities if p > 0)