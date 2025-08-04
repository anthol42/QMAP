import igraph as ig
import leidenalg
import pandas as pd

class DEF_MAPPER(dict):
    def __getitem__(self, item):
        return item

def leiden_community_detection(graph: ig.Graph, idx2id_mapping: dict = DEF_MAPPER()) -> pd.DataFrame:
    """
    Detects communities in a graph using the Leiden algorithm. It returns a dataframe containing the belonging of
    each node id to a community (Cluster).
    :param graph: The igraph.Graph object to analyze.
    :param idx2id_mapping: A dictionary mapping from the node index (an integer generated incrementally) to any id.
    :return: The dataframe containing the node ids and their corresponding cluster ids.
    """
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, n_iterations=-1)

    # Now, make dataframe with communities
    df = {"node_id": [], "community": []}
    for i, comm in enumerate(partition):
        for node in comm:
            df["node_id"].append(idx2id_mapping[node])
            df["community"].append(i)


    return pd.DataFrame(df)