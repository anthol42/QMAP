*src.qmap.toolkit.aligner*
# Function: `align_seq2db()`

```python
align_seq2db(sequences: list[str], db: VectorizedDB, batch: int = 0, device: str = "auto", index_by: typing.Literal[sequence, id] = "sequence") -> MultiAlignment:
```

**Description:** Align a list of sequences to a vectorized database and return the identity score matrix.
:param sequences: The list of sequences to align
:param db: The vectorized database that the sequences will be aligned to
:param batch: The batch number. If 0, the aligment is done in one go. Otherwise, the batch size is the number of sequences that is aligned on the full db per step
:param device: The device to use. You can use "auto" to use the available accelerator
:param index_by: The type of index to use in the multi alignment object
:return: Identity score matrix of shape (len(sequences), len(db))

**Parameters:**
- `sequences`: sequences: List[str]
- `db`: db: src.qmap.toolkit.aligner.vectorizedDB.VectorizedDB
- `batch`: batch: int = 0
- `device`: device: str = 'auto'
- `index_by`: index_by: Literal['sequence', 'id'] = 'sequence'

