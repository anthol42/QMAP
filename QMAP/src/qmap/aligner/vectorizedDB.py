import torch
from typing import Union

class VectorizedDB:
    """
    This class contains the sequences and their embeddings.
    """
    def __init__(self, sequences: list[str], embeddings: torch.Tensor):
        """
        Initialize the VectorizedDB with sequences and their embeddings.
        :param sequences: List of protein sequences.
        :param embeddings: Tensor of shape (num_sequences, sequence_length, embed_dim).
        """
        assert len(sequences) == len(embeddings), "Number of sequences must match number of embeddings."
        self.sequences = sequences
        self.embeddings = embeddings

    def embedding_by_sequence(self, sequence: str) -> torch.Tensor:
        """
        Get the embedding for a specific sequence.
        :param sequence: The protein sequence to get the embedding for.
        :return: The embedding tensor for the sequence.
        """
        index = self.sequences.index(sequence)
        if index == -1:
            raise ValueError(f"Sequence '{sequence}' not found in the database.")
        return self.embeddings[index]

    def __getitem__(self, item: Union[str, int]) -> torch.Tensor:
        if isinstance(item, int):
            return self.embeddings[item]
        elif isinstance(item, str):
            return self.embedding_by_sequence(item)
        else:
            raise ValueError(f"Unsupported item type '{type(item)}'.")

    def __len__(self) -> int:
        """
        Get the number of sequences in the database.
        :return: The number of sequences.
        """
        return len(self.sequences)

    def __iter__(self):
        for sequence, embedding in zip(self.sequences, self.embeddings):
            yield sequence, embedding

    def __str__(self):
        """
        String representation of the VectorizedDB.
        :return: A string summarizing the number of sequences and their embeddings.
        """
        s = ""
        if len(self.sequences) > 7:
            for i in range(3):
                s += f"{self.sequences[i]}: {self.embeddings[i][:8]}\n"
            s += "...\n"
            for i in range(-3, 0):
                s += f"{self.sequences[i]}: {self.embeddings[i][:8]}\n"
        else:
            for seq, emb in zip(self.sequences, self.embeddings):
                s += f"{seq}: {emb[:8]}\n"

        return s

    def __repr__(self):
        return f"VectorizedDB(sequences={len(self.sequences)}, embeddings={self.embeddings.shape})"

