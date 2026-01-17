import igraph as ig
import leidenalg
import pandas as pd
from typing import Optional

def leiden_community_detection(graph: ig.Graph, n_iterations: int = 2, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Detects communities in a graph using the Leiden algorithm. It returns a dataframe containing the belonging of
    each node id to a community (Cluster).
    :param graph: The igraph.Graph object to analyze.
    :param n_iterations: The number of iterations to run the Leiden algorithm for. If negative, it will run until
    convergence.
    :param seed: Optional seed for random number generation. If None, the random number generator will use a random seed.
    :return: The dataframe containing the node ids and their corresponding cluster ids.
    """
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, n_iterations=n_iterations, seed=seed)

    # Now, make dataframe with communities
    df = {"node_id": [], "community": []}
    for i, comm in enumerate(partition):
        for node in comm:
            df["node_id"].append(node)
            df["community"].append(i)
    return pd.DataFrame(df)