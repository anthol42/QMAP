#!/usr/bin/env python3
"""
Test script for edgelist creation with caching.
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

print("="*60)
print("Testing edgelist creation")
print("="*60)

# Get cache directory
cache_dir = pwiden_engine.get_cache_dir()
print(f"\nCache directory: {cache_dir}")

# Test 1: Create edgelist with threshold 0.3 (first run - will compute)
print("\n" + "="*60)
print("Test 1: Creating edgelist with threshold=0.3")
print("="*60)

edgelist = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.3,
    matrix="blosum62",
    gap_open=5,
    gap_extension=1,
    use_cache=True,
    show_progress=True
)

print(f"\nEdgelist shape: {edgelist.shape}")
print(f"Number of edges: {edgelist.shape[0]}")
print(f"\nEdgelist array (source_id, target_id, identity):")
print(edgelist)

# Test 2: Create same edgelist again (should load from cache)
print("\n" + "="*60)
print("Test 2: Creating same edgelist again (should use cache)")
print("="*60)

edgelist2 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.3,
    use_cache=True,
    show_progress=True
)

print(f"\nVerifying cached result matches original:")
print(f"Arrays are equal: {np.array_equal(edgelist, edgelist2)}")

# Test 3: Different threshold (will compute new)
print("\n" + "="*60)
print("Test 3: Creating edgelist with threshold=0.8")
print("="*60)

edgelist_high = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.8,
    use_cache=True,
    show_progress=True
)

print(f"\nHigh threshold edgelist shape: {edgelist_high.shape}")
print(f"Number of edges (threshold=0.8): {edgelist_high.shape[0]}")
print(f"\nEdgelist (only high similarity pairs):")
print(edgelist_high)

# Test 4: Disable cache
print("\n" + "="*60)
print("Test 4: Creating edgelist without cache")
print("="*60)

edgelist_no_cache = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.5,
    use_cache=False,
    show_progress=True
)

print(f"\nNo-cache edgelist shape: {edgelist_no_cache.shape}")
print(f"Number of edges (threshold=0.5): {edgelist_no_cache.shape[0]}")

# Test 5: Analyze edgelist structure
print("\n" + "="*60)
print("Test 5: Edgelist analysis")
print("="*60)

print(f"\nEdgelist data type: {edgelist.dtype}")
print(f"\nColumn 0 (source_id) - min: {edgelist[:, 0].min()}, max: {edgelist[:, 0].max()}")
print(f"Column 1 (target_id) - min: {edgelist[:, 1].min()}, max: {edgelist[:, 1].max()}")
print(f"Column 2 (identity) - min: {edgelist[:, 2].min():.3f}, max: {edgelist[:, 2].max():.3f}")

# Verify only upper triangle (source < target)
print(f"\nVerifying upper triangle property (all source_id < target_id):")
upper_triangle_valid = np.all(edgelist[:, 0] < edgelist[:, 1])
print(f"All edges satisfy source_id < target_id: {upper_triangle_valid}")

# Verify threshold
print(f"\nVerifying threshold property (all identity >= 0.3):")
threshold_valid = np.all(edgelist[:, 2] >= 0.3)
print(f"All edges satisfy identity >= 0.3: {threshold_valid}")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
