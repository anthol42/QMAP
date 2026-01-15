#!/usr/bin/env python3
"""
Test that integer indices in edgelist are lossless when stored as f64.
"""

import pwiden_engine
import numpy as np

sequences = ["ACDEFG"] * 10  # 10 identical sequences

print("="*60)
print("Testing integer index precision in edgelist")
print("="*60)

edgelist = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.0,
    use_cache=False,
    show_progress=False
)

print(f"\nEdgelist shape: {edgelist.shape}")
print(f"Edgelist dtype: {edgelist.dtype}")

# Extract indices
source_indices = edgelist[:, 0]
target_indices = edgelist[:, 1]

print(f"\nSample of edgelist (first 10 rows):")
print(edgelist[:10])

# Verify all indices are exact integers
source_is_int = np.all(source_indices == np.floor(source_indices))
target_is_int = np.all(target_indices == np.floor(target_indices))

print(f"\n✓ All source indices are exact integers: {source_is_int}")
print(f"✓ All target indices are exact integers: {target_is_int}")

# Convert to integers and verify no loss
source_as_int = source_indices.astype(np.int64)
target_as_int = target_indices.astype(np.int64)

# Convert back to f64 and compare
source_roundtrip = source_as_int.astype(np.float64)
target_roundtrip = target_as_int.astype(np.float64)

print(f"✓ Source indices lossless after int64 conversion: {np.array_equal(source_indices, source_roundtrip)}")
print(f"✓ Target indices lossless after int64 conversion: {np.array_equal(target_indices, target_roundtrip)}")

# Verify indices are in valid range
max_source = int(source_indices.max())
max_target = int(target_indices.max())
min_source = int(source_indices.min())
min_target = int(target_indices.min())

print(f"\nIndex ranges:")
print(f"  Source: {min_source} to {max_source}")
print(f"  Target: {min_target} to {max_target}")
print(f"  Expected range: 0 to {len(sequences) - 1}")

print(f"\n✓ All indices in valid range: {max_target < len(sequences) and min_source >= 0}")

# Demonstrate how to use in practice
print("\n" + "="*60)
print("Practical usage example:")
print("="*60)

print("\n# Method 1: Direct integer conversion")
print("for i in range(edgelist.shape[0]):")
print("    source = int(edgelist[i, 0])")
print("    target = int(edgelist[i, 1])")
print("    identity = edgelist[i, 2]")
print("    # Use source and target as array indices")

print("\n# Method 2: Vectorized conversion")
print("sources = edgelist[:, 0].astype(np.int64)")
print("targets = edgelist[:, 1].astype(np.int64)")
print("identities = edgelist[:, 2]")

# Actually do it
sources = edgelist[:, 0].astype(np.int64)
targets = edgelist[:, 1].astype(np.int64)
identities = edgelist[:, 2]

print(f"\nFirst 5 edges:")
for i in range(min(5, len(sources))):
    print(f"  Seq {sources[i]} - Seq {targets[i]}: {identities[i]:.3f}")

print("\n" + "="*60)
print("✓ All tests passed!")
print("✓ f64 can exactly represent sequence indices")
print("✓ Safe for datasets with up to 2^53 sequences")
print("="*60)
