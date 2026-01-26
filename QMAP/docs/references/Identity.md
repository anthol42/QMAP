*src.qmap.toolkit*
# Class: `Identity`

**Description:** Contains the identity matrix between all sequences in the dataset. This allows to quickly simulate
the calculation of the identity between two sequences without having to compute it on the fly.

## Method: `align_by_id()`

```python
align_by_id(self, idx1: int, idx2: int) -> float:
```

**Description:** Returns the identity between two sequences given their IDs.

