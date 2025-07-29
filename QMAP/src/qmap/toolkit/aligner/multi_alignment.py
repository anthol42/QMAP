import numpy as np


class MultiAlignment:
    def __init__(self, alignment_matrix: np.ndarray, row_sequences, col_sequences):
        """
        :param alignment_matrix: Shape(row, col)
        :param row_sequences: Sequences along the rows of the alignment matrix. [It can also be sequence ids if there are duplicates]
        :param col_sequences: Sequences along the columns of the alignment matrix. [It can also be sequence ids if there are duplicates]
        """
        self.alignment_matrix = alignment_matrix
        self.row_sequences = {str(seq): i for i, seq in enumerate(row_sequences)}
        self.col_sequences = {str(seq): i for i, seq in enumerate(col_sequences)}
        if alignment_matrix.shape[0] != len(row_sequences):
            raise ValueError("The number of rows in the alignment matrix must match the number of row sequences.")
        if alignment_matrix.shape[1] != len(col_sequences):
            raise ValueError("The number of columns in the alignment matrix must match the number of column sequences.")


    def align(self, seq1: str, seq2: str) -> str:
        """
        Align two sequences based on the alignment matrix.
        :param seq1: The first sequence to align.
        :param seq2: The second sequence to align.
        :return: The aligned sequence.
        """
        if seq1 not in self.row_sequences:
            raise ValueError(f"Sequence 1 '{seq1}' is not in the alignment matrix.")
        if seq2 not in self.col_sequences:
            raise ValueError(f"Sequence 2 '{seq2}' is not in the alignment matrix.")

        x = self.row_sequences[seq1]
        y = self.col_sequences[seq2]
        alignment = self.alignment_matrix[x, y]

        return alignment

    def __repr__(self):
        return f'MultiAlignment(nseqs1={len(self.row_sequences)}, nseqs2={len(self.col_sequences)})'