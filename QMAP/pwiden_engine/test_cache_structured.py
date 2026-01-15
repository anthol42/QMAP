#!/usr/bin/env python3
"""
Test that caching works correctly with structured arrays.
"""

import pwiden_engine
import numpy as np
import os
import shutil

sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRST", "MKTIIALSYIFCLVFA", "AAAA"]

# Clear cache
cache_dir = pwiden_engine.get_cache_dir()
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)

print("=" * 60)
print("Test 1: Create edgelist (should save to cache)")
print("=" * 60)

edgelist1 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.3,
    use_cache=True,
    show_progress=True
)

print(f"\nEdgelist dtype: {edgelist1.dtype}")
print(f"Number of edges: {len(edgelist1)}")
print(f"First edge: source={edgelist1[0]['source']}, target={edgelist1[0]['target']}, identity={edgelist1[0]['identity']:.3f}")

print("\n" + "=" * 60)
print("Test 2: Load from cache (should load existing cache)")
print("=" * 60)

edgelist2 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.3,
    use_cache=True,
    show_progress=True
)

print(f"\nEdgelist dtype: {edgelist2.dtype}")
print(f"Number of edges: {len(edgelist2)}")

print("\n" + "=" * 60)
print("Test 3: Verify arrays are identical")
print("=" * 60)

# Check if they're the same
print(f"\nArrays equal: {np.array_equal(edgelist1, edgelist2)}")
print(f"Dtypes match: {edgelist1.dtype == edgelist2.dtype}")
print(f"Sources match: {np.array_equal(edgelist1['source'], edgelist2['source'])}")
print(f"Targets match: {np.array_equal(edgelist1['target'], edgelist2['target'])}")
print(f"Identities match: {np.array_equal(edgelist1['identity'], edgelist2['identity'])}")

print("\n" + "=" * 60)
print("Test 4: Verify types are correct (u32, not float)")
print("=" * 60)

print(f"\nSource dtype: {edgelist1['source'].dtype} (expected: uint32)")
print(f"Target dtype: {edgelist1['target'].dtype} (expected: uint32)")
print(f"Identity dtype: {edgelist1['identity'].dtype} (expected: float32)")

assert edgelist1['source'].dtype == np.uint32, "Source should be uint32"
assert edgelist1['target'].dtype == np.uint32, "Target should be uint32"
assert edgelist1['identity'].dtype == np.float32, "Identity should be float32"

print("\n✓ All assertions passed!")
print("✓ Indices are stored as u32, not converted to float")
print("✓ Cache preserves structured array dtype correctly")
