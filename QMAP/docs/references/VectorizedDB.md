*src.qmap.toolkit.aligner*
# Class: `VectorizedDB`

**Description:** This class contains the sequences and their embeddings.

## Method: `embedding_by_id()`

```python
embedding_by_id(self, id: str) -> Tensor:
```

**Description:** Get the embedding for a specific sequence id. This method will work only if you provided ids when encoding the sequences.

**Parameters:**
- `id`: The sequence id to get the embedding for.

**Return:**
- The embedding tensor for the sequence id.
## Method: `embedding_by_sequence()`

```python
embedding_by_sequence(self, sequence: str) -> Tensor:
```

**Description:** Get the embedding for a specific sequence.

**Parameters:**
- `sequence`: The protein sequence to get the embedding for.

**Return:**
- The embedding tensor for the sequence.
