*src.qmap.toolkit.clustering*
# Function: `build_graph()`

```python
build_graph(db: VectorizedDB, threshold: float, batch_size: int = 0, path: typing.Union[str, NoneType] = None) -> tuple[str, dict]:
```

**Description:** Builds a graph from a VectorDB of sequences
:param db: The VectorDB to build the graph from
:param threshold: The threshold to establish an edge between two sequences based on their identity score
:param batch_size: The number of sequences to process at each step. If 0, the whole db is processed at once.
:param path: The path to save the graph to. If None, the graph is saved in a temporary folder.
:return: The path to the edgelist file containing the graph. The file is contained in the tmp folder if not specified.

**Parameters:**
- `db`: db: qmap.toolkit.aligner.vectorizedDB.VectorizedDB
- `threshold`: threshold: float
- `batch_size`: batch_size: int = 0
- `path`: path: Optional[str] = None

