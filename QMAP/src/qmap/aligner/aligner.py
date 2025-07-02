from .vectorizedDB import VectorizedDB
from .encoder import Encoder
import numpy as np
from .utils import _get_device
import torch
from pyutils import progress
from typing import List


def _batch_align(db1: VectorizedDB, db2: VectorizedDB, batch: int) -> torch.Tensor:
    """
    Align two vectorized databases in batches and return the identity score matrix.
    :param db1: The vectorized database to align on the other db2
    :param db2: The vectorized database that the db1 will be aligned to
    :param batch: The batch size. The number of sequences of db1 that is aligned on the full db2 per step
    :return: Identity score matrix of shape (len(db1), len(db2))
    """
    device = _get_device()
    iden = torch.zeros((len(db1), len(db2)), device=device)
    db2_emb = db2.embeddings.T.to(device)
    for i in progress(range(0, len(db1), batch), type="pip", desc="Aligning databases"):
        end = min(i + batch, len(db1))
        iden[i:end] = (db1.embeddings[i:end].to(device) @ db2_emb).cpu()

    return iden


def align_db(db1: VectorizedDB, db2: VectorizedDB, batch: int = 0, device: str = "auto") -> np.ndarray:
    """
    Align two vectorized databases and return the identity score matrix. (pseudo identities)
    :param db1: The vectorized database to align on the other db2
    :param db2: The vectorized database that the db1 will be aligned to
    :param batch: The batch number. If 0, the aligment is done in one go. Otherwise, the batch size is the number of sequences of db1 that is aligned on the full db2 per step
    :param device: The device to use. You can use "auto" to use the available accelerator
    :return: Identity score matrix of shape (len(db1), len(db2))
    """
    device = _get_device() if device == "auto" else device
    if batch > 0:
        iden = _batch_align(db1, db2, batch)
    else:
        iden = (db1.embeddings.to(device) @ db2.embeddings.T.to(device)).cpu()

    return iden.numpy()

def align_seq2db(sequences: List[str], db: VectorizedDB, batch: int = 0, device: str = "auto") -> np.ndarray:
    """
    Align a list of sequences to a vectorized database and return the identity score matrix.
    :param sequences: The list of sequences to align
    :param db: The vectorized database that the sequences will be aligned to
    :param batch: The batch number. If 0, the aligment is done in one go. Otherwise, the batch size is the number of sequences that is aligned on the full db per step
    :param device: The device to use. You can use "auto" to use the available accelerator
    :return: Identity score matrix of shape (len(sequences), len(db))
    """
    device = _get_device() if device == "auto" else device

    # Create a VectorizedDB from the sequences
    encoder = Encoder()
    seq_db = encoder.encode(sequences)

    return align_db(seq_db, db, batch=batch, device=device)

def align_seq(seq1: str, seq2: str):
    """
    Align two sequences and return the identity score.
    :param seq1: The first sequence
    :param seq2: The second sequence
    :return: The identity score between the two sequences
    """
    encoder = Encoder()
    db1 = encoder.encode([seq1])
    db2 = encoder.encode([seq2])

    iden = align_db(db1, db2, batch=0)

    return iden[0, 0]