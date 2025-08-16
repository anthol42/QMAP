*src.qmap.toolkit.split*
# Function: `filter_out()`

```python
filter_out(train_sequences: list[str], metadata: list[Any], ref_sequences: list[str], threshold: float = 0.55, encoder_batch_size: int = 512, aligner_batch_size: int = 0) -> tuple:
```

**Description:** Removes samples in train_sequences and metadata that are more similar than the threshold to any sequence in
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
:return: The filtered train_sequences and metadata.

**Parameters:**
- `train_sequences`: train_sequences: List[str]
- `metadata`: *metadata: List[Any]
- `ref_sequences`: ref_sequences: List[str]
- `threshold`: threshold: float = 0.55
- `encoder_batch_size`: encoder_batch_size: int = 512
- `aligner_batch_size`: aligner_batch_size: int = 0

