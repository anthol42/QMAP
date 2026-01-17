from pwiden_engine import compute_global_identity, create_edgelist
import numpy as np
import pytest


def test_edgelist_matches_matrix():
    """
    Test that create_edgelist produces the same edges as extracting from
    the global identity matrix with the same threshold.
    """
    # Load sequences from file
    with open("test_sequences.txt", "r") as f:
        sequences = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(sequences)} sequences")

    # Set threshold for edge filtering
    threshold = 0.5

    # Method 1: Compute global identity matrix
    print("Computing global identity matrix...")
    identity_matrix = compute_global_identity(sequences, show_progress=True)

    # Extract edges from matrix where identity > threshold
    print(f"Extracting edges from matrix with threshold {threshold}...")
    expected_edges = {}
    n = len(sequences)
    for source in range(n):
        for target in range(source + 1, n):  # Only upper triangle, source < target
            if identity_matrix[source, target] >= threshold:
                expected_edges[(source, target)] = identity_matrix[source, target]

    print(f"Found {len(expected_edges)} edges from matrix")

    # Method 2: Use create_edgelist directly
    print("Creating edgelist...")
    edgelist_dict = create_edgelist(
        sequences,
        threshold=threshold,
        show_progress=True,
        use_cache=False  # Disable cache to ensure fresh computation
    )

    print(f"Found {len(edgelist_dict)} edges from create_edgelist")

    assert edgelist_dict == expected_edges, "Edgelist from create_edgelist does not match expected edges from matrix"

    # Convert edgelist to same format for comparison
    unmatched_edges = []
    for key in expected_edges:
        if key not in edgelist_dict:
            unmatched_edges.append(key)
        else:
            if not np.isclose(expected_edges[key], edgelist_dict[key]):
                unmatched_edges.append(key)



    # Verify same number of edges
    assert len(unmatched_edges) == 0, f"Unmatched edges found in create_edgelist: {unmatched_edges}"

    print("✓ All edges match between matrix and edgelist methods")


if __name__ == "__main__":
    test_edgelist_matches_matrix()