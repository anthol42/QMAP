*src.qmap.toolkit.aligner*
# Function: `align_db()`

```python
align_db(db1: VectorizedDB, db2: VectorizedDB, batch: int = 0, device: str = "auto", index_by: typing.Literal[sequence, id] = "sequence") -> MultiAlignment:
```

**Description:** Align two vectorized databases and return the identity score matrix. (pseudo identities)
:param db1: The vectorized database to align on the other db2
:param db2: The vectorized database that the db1 will be aligned to
:param batch: The batch number. If 0, the aligment is done in one go. Otherwise, the batch size is the number of sequences of db1 that is aligned on the full db2 per step
:param device: The device to use. You can use "auto" to use the available accelerator
:param index_by: The type of index to use in the multi alignment object
:return: Identity score matrix of shape (len(db1), len(db2))

**Parameters:**
- `db1`: db1: src.qmap.toolkit.aligner.vectorizedDB.VectorizedDB
- `db2`: db2: src.qmap.toolkit.aligner.vectorizedDB.VectorizedDB
- `batch`: batch: int = 0
- `device`: device: str = 'auto'
- `index_by`: index_by: Literal['sequence', 'id'] = 'sequence'

