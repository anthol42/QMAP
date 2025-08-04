import pytest
import torch
from torch import testing as torchtest
from qmap.toolkit.aligner import Encoder, MultiAlignment
from qmap.toolkit.aligner import aligner
import numpy as np
import numpy.testing as npt

@pytest.fixture
def sequences():
    return [
        "DSHAKRHHGYKRKFHEKHHSHRGY", # Normal sequence
        "KVVVKWVVKVVK",  # Low complexity sequence
        "YVLLKRKRLIFI", # Synthetic sequence
        "VVVVVV", # No complexity sequence
        "VRNHVTCRINRGFCVPIRCPGRTRQIGTCFGPRIKCCRSW", # Normal long sequence
        "LXXVA", # Contains X
        "K", # len = 1
        "SYQRIRSDHDSHSCANNRGWCRPTCFSHEYTDWFNNDVCGSYRCCRPGRRPRTYULLAVAGGHNEEEGHTURVLILIAVEGHRLRGAVLPPEPEPIHKRL" # len = 100
    ]

@pytest.fixture
def encoder():
    return Encoder(force_cpu=True)

def test_encoder_encode_no_batch_no_ids(sequences, encoder):
    """
    Test encoding with a batch size bigger than the number of sequences and no ids.
    """
    vectorizedDB = encoder.encode(sequences)

    expected_embeddings = torch.load("assets/expected_embeddings.pt")

    # Assert embeddings match expected values
    torchtest.assert_close(vectorizedDB.embeddings, expected_embeddings)
    # Assert sequence order is preserved
    assert vectorizedDB.sequences == sequences


def test_encoder_encode_batch_no_ids(sequences, encoder):
    """
    Test encoding with a batch size smaller than the number of sequences and no ids.
    """
    vectorizedDB = encoder.encode(sequences, batch_size=2) # 4 steps

    expected_embeddings: torch.Tensor = torch.load("assets/expected_embeddings.pt")
    # Assert embeddings match expected values
    torchtest.assert_close(vectorizedDB.embeddings, expected_embeddings)
    # Assert sequence order is preserved
    assert vectorizedDB.sequences == sequences

def test_encoder_encode_batch_with_ids(sequences, encoder):
    """
    Test encoding with a batch size smaller than the number of sequences and with ids.
    """
    ids = [f"seq_{i}" for i in range(len(sequences))]
    vectorizedDB = encoder.encode(sequences, batch_size=2, ids=ids) # 4 steps

    expected_embeddings = torch.load("assets/expected_embeddings.pt")
    # Assert embeddings match expected values
    torchtest.assert_close(vectorizedDB.embeddings, expected_embeddings)
    # Assert sequence order is preserved
    assert vectorizedDB.sequences == sequences
    # Assert ids are preserved
    assert vectorizedDB.ids == ids

def test_multi_alignment(sequences):
    alignment_matrix = np.random.rand(len(sequences), len(sequences))
    row_seq = sequences
    col_seq = list(reversed(sequences))  # Reverse the sequences for columns
    ma = MultiAlignment(
        alignment_matrix=alignment_matrix,
        row_sequences=row_seq,
        col_sequences=col_seq,
    )
    # Test align method
    expected = alignment_matrix[2, 3]
    result = ma.align(row_seq[2], col_seq[3])
    assert result == expected

def test_align_db_sequences(sequences, encoder):
    """
    Test aligning two vectorized databases together with sequences as ids.
    """
    db = encoder.encode(sequences)
    matrix = aligner.align_db(db, db)

    expected = np.load("assets/expected_alignment_matrix.npy")
    assert isinstance(matrix, MultiAlignment)
    assert np.allclose(matrix.alignment_matrix, expected)



def test_align_db_ids(sequences, encoder):
    """
    Test aligning two vectorized databases together with ids as ids.
    """
    ids = [f"seq_{i}" for i in range(len(sequences))]
    db = encoder.encode(sequences, ids=ids)
    matrix = aligner.align_db(db, db, index_by="id")

    pred_align = matrix.align("seq_2", "seq_3")
    expected = np.load("assets/expected_alignment_matrix.npy")[2, 3]
    assert pred_align == expected

def test_batch_align(sequences, encoder):
    db = encoder.encode(sequences)
    matrix = aligner.align_db(db, db, batch=2) # Step: 4

    expected = np.load("assets/expected_alignment_matrix.npy")
    assert isinstance(matrix, MultiAlignment)
    assert np.allclose(matrix.alignment_matrix, expected)
