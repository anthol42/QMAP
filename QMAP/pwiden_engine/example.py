"""
Example usage of pwiden_engine for pairwise sequence identity calculations.
"""

import pwiden_engine
import numpy as np

# Example protein sequences
sequences = [
    "ACDEFGHIKLMNPQRSTVWY",  # All 20 amino acids
    "ACDEFGHIKLMNPQRSTVWY",  # Identical to first
    "ACDEFGHIKLMNPQRST",     # Shorter, similar to first
    "VWXYZACDEFG",           # Different sequence
    "MKTIIALSYIFCLVFA",      # Random peptide
]

print("Computing pairwise identity matrix...")
print(f"Number of sequences: {len(sequences)}\n")

# Compute global alignment
identity_matrix = pwiden_engine.compute_global_identity(
    sequences,
    matrix="blosum62",
    gap_open=5,
    gap_extension=1,
    show_progress=True
)

print("\nGlobal alignment identity matrix:")
print(identity_matrix)
print(f"Shape: {identity_matrix.shape}")
print(f"Type: {type(identity_matrix)}")

# Verify matrix properties
print("\n" + "="*60)
print("Matrix properties:")
print(f"Mean identity: {identity_matrix.mean():.3f}")
print(f"Max identity: {identity_matrix.max():.3f}")
print(f"Min identity: {identity_matrix.min():.3f}")

# Verify diagonal is all 1.0
print("\n" + "="*60)
print("Diagonal values (should all be 1.0):")
print(np.diag(identity_matrix))
print(f"All diagonal values are 1.0: {np.allclose(np.diag(identity_matrix), 1.0)}")

# Verify matrix is symmetric
print("\n" + "="*60)
print("Matrix symmetry check:")
is_symmetric = np.allclose(identity_matrix, identity_matrix.T)
print(f"Matrix is symmetric: {is_symmetric}")
if is_symmetric:
    print("✓ Upper triangle successfully copied to lower triangle")

# Show some example comparisons
print("\n" + "="*60)
print("Example comparisons:")
print(f"Seq 0 vs Seq 1 (identical): {identity_matrix[0, 1]:.3f}")
print(f"Seq 0 vs Seq 2 (similar):   {identity_matrix[0, 2]:.3f}")
print(f"Seq 0 vs Seq 3 (different): {identity_matrix[0, 3]:.3f}")
print(f"Seq 2 vs Seq 0 (symmetric): {identity_matrix[2, 0]:.3f}")

# Save to file
np.save("identity_matrix.npy", identity_matrix)
print("\n" + "="*60)
print("Saved identity matrix to identity_matrix.npy")
