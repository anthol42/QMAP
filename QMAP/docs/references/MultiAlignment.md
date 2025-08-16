*src.qmap.toolkit.aligner*
# Class: `MultiAlignment`

**Description:** Wraps the alignment matrix of two sets of sequences. You can access the alignment matrix directly from the
`alignment_matrix` attribute. You can also simulate an alignment between two sequences using the `align` method.
this will return the precomputed alignment between the two sequences.

## Method: `align()`

```python
align(self, seq1: str, seq2: str) -> str:
```

**Description:** Align two sequences based on the alignment matrix. The parameters can be the sequences itself or the sequence id
depending on how the class was initialized. Using the default align functions, you can choose whether to index
by index or sequence with the `index_by` parameter.

**Parameters:**
- `seq1`: The first sequence to align.
- `seq2`: The second sequence to align.

**Return:**
- The aligned sequence.
