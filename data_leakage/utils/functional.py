import json
from pyutils import progress
import torch
from transformers import EsmTokenizer, EsmModel
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices

def read_fasta(file_path):
    """
    Reads a FASTA file and returns a list of tuples containing sequence IDs and sequences.
    :param file_path: The path to the FASTA file.
    :return:
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sequences = []
    for line in lines:
        if line.startswith('>'):
            id_ = line[1:].strip()
        else:
            sequence = line.strip()
            sequences.append((id_, sequence))
    return sequences

def read_json(file_path):
    """
    Reads a JSON file and returns the parsed data.
    :param file_path: The path to the JSON file.
    :return:
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def generate_esm2_embeddings(protein_sequences: list[str],
                             model_name: str = "facebook/esm2_t33_650M_UR50D",
                             device: str = None,
                             batch_size: int = 512) -> np.ndarray:
    """
    Generate protein embeddings using ESM2 model with mean pooling.

    Args:
        protein_sequences (List[str]): List of protein sequences (amino acid sequences)
        model_name (str): Name of the ESM2 model to use. Options include:
            - "facebook/esm2_t6_8M_UR50D" (8M parameters)
            - "facebook/esm2_t12_35M_UR50D" (35M parameters)
            - "facebook/esm2_t30_150M_UR50D" (150M parameters)
            - "facebook/esm2_t33_650M_UR50D" (650M parameters) - default
            - "facebook/esm2_t36_3B_UR50D" (3B parameters)
            - "facebook/esm2_t48_15B_UR50D" (15B parameters)
        device (str): Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        batch_size (int): Batch size for processing sequences

    Returns:
        np.ndarray: Array of shape (num_sequences, embedding_dim) containing embeddings
                   in the same order as input sequences
    """

    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer and model
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []

    # Process sequences in batches
    for i in progress(range(0, len(protein_sequences), batch_size)):
        batch_sequences = protein_sequences[i:i + batch_size]

        # Tokenize sequences
        inputs = tokenizer(batch_sequences,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=1024)

        # Move to device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Process each sequence in the batch
        for j in range(len(batch_sequences)):
            # Get attention mask for this sequence (to exclude padding tokens)
            attention_mask = inputs['attention_mask'][j]
            seq_embeddings = hidden_states[j][attention_mask.bool()]  # Remove padding

            # Remove special tokens (CLS and EOS tokens)
            seq_embeddings_no_special = seq_embeddings[1:-1]  # Remove first (CLS) and last (EOS) tokens

            # Mean pooling
            seq_embedding = seq_embeddings_no_special.mean(dim=0)
            all_embeddings.append(seq_embedding.cpu().numpy())

    return np.array(all_embeddings)

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