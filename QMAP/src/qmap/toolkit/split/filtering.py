from typing import List, Any
from ..aligner import aligner
from ..aligner import Encoder


def filter_out(train_sequences: List[str], *metadata: List[Any], ref_sequences: List[str],
               threshold: float = 0.55, encoder_batch_size: int = 512, aligner_batch_size: int = 0, force_cpu: bool = True) -> tuple:
    """
    Removes samples in train_sequences and metadata that are more similar than the threshold to any sequence in
    ref_sequences.

    ## Example
    ```python
    from qmap.toolkit.split import filter_out

    train_seq, train_labels = filter_out(train_sequences, train_labels, ref_sequences=test_sequences, threshold=0.55)
    ```
    :param train_sequences: The sequences that will be filtered.
    :param metadata: The metadata associated with the sequences that will be filtered.
    :param ref_sequences: The reference sequences, usually the test set sequences or the benchmark sequences.
    :param threshold: The threshold above which the sequences are considered similar and should be filtered out.
    :param encoder_batch_size: The batch size to use for encoding the sequences. Reduce it if you get an out of memory error.
    :param aligner_batch_size: The batch size to use for the aligner. If set to 0, it will use the full dataset size.
    :param force_cpu: If true, it will encode the sequences on the CPU.
    :return: The filtered train_sequences and metadata.
    """
    encoder = Encoder(force_cpu=force_cpu)
    ref_db = encoder.encode(ref_sequences, ids=[f'ref_{i}' for i in range(len(ref_sequences))], batch_size=encoder_batch_size)
    train_db = encoder.encode(train_sequences, ids=[f'train_{i}' for i in range(len(train_sequences))], batch_size=encoder_batch_size)
    iden_scores = aligner.align_db(train_db, ref_db, batch=aligner_batch_size, index_by='id')
    graph = iden_scores.alignment_matrix > threshold
    should_remove = graph.any(axis=1)
    filtered_train_sequences = [seq for i, seq in enumerate(train_sequences) if not should_remove[i]]
    filtered_metadata = [[item for i, item in enumerate(meta) if not should_remove[i]] for meta in metadata]

    if len(metadata) == 0:
        return filtered_train_sequences,
    else:
        return filtered_train_sequences, *filtered_metadata
