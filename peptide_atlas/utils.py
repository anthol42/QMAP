from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
import numpy as np

def create_blosum_with_xo(matrix_name="BLOSUM62", x_penalty=-1, x_self_score=0, o_penalty=-2, o_self_score=2):
    """
    Create a BLOSUM matrix that handles both X and O residues
    """
    original_matrix = substitution_matrices.load(matrix_name)
    original_alphabet = list(original_matrix.alphabet)

    # Add X and O if not present
    extended_alphabet = original_alphabet.copy()
    if 'X' not in extended_alphabet:
        extended_alphabet.append('X')
    if 'O' not in extended_alphabet:
        extended_alphabet.append('O')

    custom_matrix = substitution_matrices.Array(alphabet=tuple(extended_alphabet), dims=2)

    # Copy original scores
    for aa1 in original_alphabet:
        for aa2 in original_alphabet:
            custom_matrix[aa1, aa2] = original_matrix[aa1, aa2]

    # Handle X
    for aa in extended_alphabet:
        if aa != 'X':
            custom_matrix['X', aa] = x_penalty
            custom_matrix[aa, 'X'] = x_penalty
    custom_matrix['X', 'X'] = x_self_score

    # Handle O (pyrrolysine)
    # Pyrrolysine is chemically similar to lysine, so give it reasonable scores
    for aa in extended_alphabet:
        if aa == 'K':  # Lysine - most similar
            custom_matrix['O', aa] = 1
            custom_matrix[aa, 'O'] = 1
        elif aa == 'R':  # Arginine - also basic
            custom_matrix['O', aa] = 0
            custom_matrix[aa, 'O'] = 0
        elif aa == 'O':  # O vs O
            custom_matrix['O', aa] = o_self_score
        elif aa == 'X':  # Already handled above
            continue
        else:  # All other amino acids
            custom_matrix['O', aa] = o_penalty
            custom_matrix[aa, 'O'] = o_penalty

    return custom_matrix

def compute_identity_from_aligned(s1: str, s2: str) -> float:
    """
    Compute the identity between two sequences. (Score between 0 and 1)
    :param s1: The first sequence.
    :param s2: The second sequence.
    :return: The identity score between 0 and 1.
    """
    matches = 0
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        if a != '-' and b != '-':
            if a == b:
                matches += 1
    return matches / len(s1)

def compute_identity(s1: str, s2: str) -> float:
    """
    Compute the identity between two sequences using a pairwise aligner. (Score between 0 and 1)
    :param s1: The first sequence.
    :param s2: The second sequence.
    :return: The similarity score between 0 and 1.
    """
    aligner = PairwiseAligner()
    aligner.substitution_matrix = create_blosum_with_xo('BLOSUM45')
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    standard_aa = set('ACDEFGHIKLMNPQRSTVWYXO')
    invalid_chars = set(s2.upper().strip()) - standard_aa
    if invalid_chars:
        print(s2.upper().strip())
        print(f"Invalid characters found: {invalid_chars}")
    alignments = aligner.align(s1.upper().strip(), s2.upper().strip())
    best = alignments[0]
    return compute_identity_from_aligned(best[0], best[1])

def read_fasta(file_path):
    """
    Reads a FASTA file and returns a list of tuples containing sequence IDs and sequences.
    :param file_path: The path to the FASTA file.
    :return:
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sequences = {}
    for line in lines:
        if line.startswith('>'):
            id_ = int(line[1:].strip().replace("seq_", ""))
        else:
            sequence = line.strip()
            sequences[id_] = sequence
    return sequences

def sequence_entropy(sequence: str) -> float:
    """
    Calculate the Shannon entropy of a sequence.
    :param sequence: The sequence to calculate the entropy for
    :return: The Shannon entropy of the sequence
    """
    from collections import Counter
    from math import log2

    counts = Counter(sequence)
    total = len(sequence)
    probabilities = [count / total for count in counts.values()]
    return -sum(p * log2(p) for p in probabilities if p > 0)

def _low_complexity_sequence(max_motif_length: int = 4, max_length: int = 100,
                                      mutation_rate: float = 0.05):
    """
    Generate low complexity sequences.
    :return: A generator yielding low complexity sequences.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    # Step 1: Choose the motif length
    while True:
        motif_length = np.random.randint(1, max_motif_length)
        if motif_length == 1 and np.random.rand() < 0.95:  # 5% chance to have a motif length of 1
            continue
        break
    motif = ''.join(np.random.choice(list(amino_acids), size=motif_length))

    # Step 2: Choose the number of repetitions
    max_motif = max_length // motif_length
    # The repetition follows a geometric distribution

    repetitions = np.random.geometric(0.25) + 1  # +1 to ensure at least two repetition
    repetitions = min(repetitions, max_motif)

    # Step 3: Generate the sequence
    sequence = motif * repetitions

    # Step 4: Apply random mutations
    sequence = list(sequence)
    for i in range(len(sequence)):
        if np.random.rand() < mutation_rate:
            sequence[i] = np.random.choice(list(amino_acids))
    sequence = ''.join(sequence)
    return sequence

def low_complexity_sequence_generator(num_subsequences: int = 1, max_motif_length: int = 4, max_length: int = 100,
                                        mutation_rate: float = 0.025, remove_anomalies: bool = True):
    """
    Generate low complexity sequences.
    :param num_subsequences: The number of subsequences to generate within the sequence. Samples randomly between 1 and num_subsequences.
    :param max_motif_length: The max length of the motif to repeat. Samples randomly between 1 and max_motif_length. However, a motif length of 1 is unlikely (1% of 1/max_motif_length)
    :param max_length: The maximum length of the sequence.
    :param mutation_rate: The probability to induce a mutation in each amino acid of the sequence.
    :param remove_anomalies: Remove sequences that have a complexity of 1 or 1.584962500721156
    :return: yield the sequences
    """
    while True:
        num_subsequence = np.random.randint(1, num_subsequences + 1) if num_subsequences > 1 else 1
        # print(num_subsequence)
        seq = ""
        for _ in range(num_subsequence):
            part = _low_complexity_sequence(max_motif_length, max_length // num_subsequences, mutation_rate)
            if remove_anomalies:
                while (sequence_entropy(part) == 1 or abs(sequence_entropy(part) -  1.584962500721156) < 1e-6  or
                       abs(sequence_entropy(part) - 0.9182958340544896) < 1e-6):
                    part = _low_complexity_sequence(max_motif_length, max_length // num_subsequences, mutation_rate)
            seq += part

        yield seq

def high_complexity_sequence_generator(max_length: int = 100):
    """
    Generate high complexity sequences.
    :param max_length: The maximum length of the sequence.
    :return: yield the sequences
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    while True:
        length = np.random.binomial(max_length, 0.25)
        if length > max_length:
            length = max_length
        seq = ''.join(np.random.choice(list(amino_acids), size=length))
        yield seq