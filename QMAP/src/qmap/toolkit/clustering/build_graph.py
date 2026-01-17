from qmap.toolkit.aligner import create_edgelist
import igraph as ig

def build_graph(sequences: list[str], threshold: float, *args, **kwargs) -> tuple[ig.Graph, dict[tuple[int, int], float]]:
    """
    Builds a graph from a list of sequences and an identity threshold. Sequences are connected if their identity score
    is above or equal to the threshold.
    :param sequences: The list of sequences to build the graph from
    :param threshold: The threshold to establish an edge between two sequences based on their identity score
    :param args: Additional positional arguments to pass to the pwiden engine that will create the edges.
    :param kwargs: Additional keyword arguments to pass to the pwiden engine that will create the edges.
    :return: The created graph and the edgelist dictionary (source_node, target_node) -> identity_score
    """
    edgelist = create_edgelist(sequences, threshold, *args, **kwargs)
    # return ig.Graph.TupleList(edgelist.keys(), directed=False), edgelist
    return ig.Graph(n=len(sequences), edges=list(edgelist.keys()), directed=False), edgelist
