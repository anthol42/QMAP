from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices

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