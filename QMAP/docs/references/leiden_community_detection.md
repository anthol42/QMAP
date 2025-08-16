*src.qmap.toolkit.clustering*
# Function: `leiden_community_detection()`

```python
leiden_community_detection(graph: Graph, idx2id_mapping: dict = {}, n_iterations: int = -1) -> DataFrame:
```

**Description:** Detects communities in a graph using the Leiden algorithm. It returns a dataframe containing the belonging of
each node id to a community (Cluster).
:param graph: The igraph.Graph object to analyze.
:param idx2id_mapping: A dictionary mapping from the node index (an integer generated incrementally) to any id.
:return: The dataframe containing the node ids and their corresponding cluster ids.

**Parameters:**
- `graph`: graph: igraph.Graph
- `idx2id_mapping`: idx2id_mapping: dict = {}
- `n_iterations`: n_iterations: int = -1

