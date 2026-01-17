from qmap.toolkit.aligner import create_edgelist
import igraph as ig

def build_graph(sequences: list[str], threshold: float, *args, **kwargs) -> ig.Graph:
    """
    Builds a graph from a list of sequences and an identity threshold. Sequences are connected if their identity score
    is above or equal to the threshold.
    :param sequences: The list of sequences to build the graph from
    :param threshold: The threshold to establish an edge between two sequences based on their identity score
    :param args: Additional positional arguments to pass to the pwiden engine that will create the edges.
    :param kwargs: Additional keyword arguments to pass to the pwiden engine that will create the edges.
    :return: The path to the edgelist file containing the graph. The file is contained in the tmp folder if not specified.
    """
    edgelist = create_edgelist(sequences, threshold, *args, **kwargs)
    return ig.Graph.TupleList(edgelist.keys(), directed=False)
