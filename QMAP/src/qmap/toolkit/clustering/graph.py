"""
Builds a edgelist graph file from a VectorDB and an identity threshold
"""
import torch
from typing import Optional, Tuple
import numpy as np
from qmap.toolkit.aligner import VectorizedDB, Encoder
import tempfile
import uuid

def build_graph(db: VectorizedDB, threshold: float, batch_size: int = 0, path: Optional[str] = None) -> Tuple[str, dict]:
    """
    Builds a graph from a VectorDB of sequences
    :param db: The VectorDB to build the graph from
    :param threshold: The threshold to establish an edge between two sequences based on their identity score
    :param batch_size: The number of sequences to process at each step. If 0, the whole db is processed at once.
    :param path: The path to save the graph to. If None, the graph is saved in a temporary folder.
    :return: The path to the edgelist file containing the graph. The file is contained in the tmp folder if not specified.
    """
    activation = Encoder(force_cpu=True).activation
    activation = activation.half()
    if path is None:
        path = f'{tempfile.gettempdir()}/qmap_graph_{uuid.uuid4()}.edgelist'

    if batch_size == 0:
        batch_size = len(db)
    if db.ids:
        idx2id = {i: id_ for i, id_ in enumerate(db.ids)}
    else:
        idx2id = {i: i for i in range(len(db))}
    for i in range(0, len(db), batch_size):
        end = min(i + batch_size, len(db))
        batch = db.embeddings[i:end]
        with torch.inference_mode():
            iden = activation(batch @ db.embeddings.T).numpy()
        adjacency = iden > threshold
        node_ids = i + np.arange(len(iden))

        batch_indices, target_indices = np.where(adjacency)
        source_nodes = node_ids[batch_indices]
        edges = np.column_stack([source_nodes, target_indices])

        # Append to file
        with open(path, 'a') as f:
            np.savetxt(f, edges, fmt='%d %d')

    return path, idx2id
