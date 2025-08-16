*src.qmap.toolkit.aligner*
# Class: `Encoder`

**Description:** This class is used to encode the protein sequences into embeddings. It will return a VectorizedDB object which is
mainly a wrapper of a tensor of shape (N_sequences, 512) corresponding to the sequence embeddings.

Usage:

Its usage is very simple, simply initialize the class and call the encode method with a list of sequences!

## Method: `encode()`

```python
encode(self, sequences: list[str], batch_size: int = 512, ids: typing.Union[list[str], NoneType] = None) -> VectorizedDB:
```

**Description:** Encode a list of sequences using the model.

**Parameters:**
- `sequences`: List of peptide sequences to encode. Note that the sequences should have a maximum length of 100 amino acids. Please filter out longer sequences or truncate them before encoding.
- `ids`: The sequence ids. Useful when the sequences are not unique. If not provided, you can still find the sequence by its index as the order of the embeddings is the same as the order of the sequences.
- `batch_size`: The batch size to use. Change this to a lower value if you run out of memory.

**Return:**
- Encoded tensor of shape (N_sequences, 512).
