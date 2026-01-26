*src.qmap.toolkit*
# Function: `leiden_community_detection()`

```python
leiden_community_detection(graph: Graph, n_iterations: int = 2, seed: typing.Union[int, NoneType] = None) -> DataFrame:
```

**Description:** Detects communities in a graph using the Leiden algorithm. It returns a dataframe containing the belonging of
each node id to a community (Cluster).
:param graph: The igraph.Graph object to analyze.
:param n_iterations: The number of iterations to run the Leiden algorithm for. If negative, it will run until
convergence.
:param seed: Optional seed for random number generation. If None, the random number generator will use a random seed.
:return: The dataframe containing the node ids and their corresponding cluster ids.

**Parameters:**
- `graph`: graph: igraph.Graph
- `n_iterations`: n_iterations: int = 2
- `seed`: seed: Optional[int] = None

