import pytest
from qmap.toolkit import clustering
import numpy as np
import numpy.testing as npt

def _load_edgelist(path) -> np.ndarray:
    """
    Make an adjacency matrix from an edgelist file.
    :param path: The path to the edgelist file.
    :return: The adjacency matrix.
    """
    # Read the edgelist file
    edges = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        source = int(parts[0])
                        target = int(parts[1])
                        edges.append((source, target))
                    except ValueError:
                        continue  # Skip lines that don't have valid integers

    if not edges:
        return np.array([])

    # Find the maximum node ID to determine matrix size
    all_nodes = set()
    for source, target in edges:
        all_nodes.add(source)
        all_nodes.add(target)

    max_node = max(all_nodes)
    min_node = min(all_nodes)

    # Create adjacency matrix (assuming 0-indexed or adjust if needed)
    if min_node == 0:
        matrix_size = max_node + 1
        node_offset = 0
    else:
        # If nodes don't start at 0, you might want to remap them
        # This version assumes consecutive numbering starting from min_node
        matrix_size = max_node - min_node + 1
        node_offset = min_node

    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Fill the adjacency matrix
    for source, target in edges:
        adj_matrix[source - node_offset][target - node_offset] = 1
        # Uncomment the next line if the graph is undirected
        # adj_matrix[target - node_offset][source - node_offset] = 1

    return adj_matrix

def test_build_graph_no_batch(sequences, encoder):
    """
    Test the build_graph function
    """
    ids = [f"seq_{i}" for i in range(len(sequences))]
    db = encoder.encode(sequences, ids=ids)

    path, idx2id = clustering.build_graph(db, threshold=0.55)
    assert path is not None
    expected_mapper = {i: f"seq_{i}" for i in range(len(sequences))}
    assert expected_mapper == idx2id

    pred_adj_matrix = _load_edgelist(path)
    expected = (np.load("assets/expected_alignment_matrix.npy") > 0.55).astype(int)
    npt.assert_array_equal(pred_adj_matrix, expected)


def test_build_graph_batch(sequences, encoder):
    """
    Test the build_graph function with a batch size smaller than the number of sequences.
    """
    ids = [f"seq_{i}" for i in range(len(sequences))]
    db = encoder.encode(sequences, ids=ids)

    path, idx2id = clustering.build_graph(db, threshold=0.55, batch_size=2)
    assert path is not None
    expected_mapper = {i: f"seq_{i}" for i in range(len(sequences))}
    assert expected_mapper == idx2id

    pred_adj_matrix = _load_edgelist(path)
    expected = (np.load("assets/expected_alignment_matrix.npy") > 0.55).astype(int)
    npt.assert_array_equal(pred_adj_matrix, expected)