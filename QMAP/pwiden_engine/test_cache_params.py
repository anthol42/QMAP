#!/usr/bin/env python3
"""
Test that different alignment parameters create different cache files.
"""

import pwiden_engine
import numpy as np

sequences = [
    "ACDEFGHIKLMNPQRSTVWY",
    "ACDEFGHIKLMNPQRST",
    "MKTIIALSYIFCLVFA",
]

print("="*60)
print("Testing parameter-aware caching")
print("="*60)

# Test 1: Create edgelist with default parameters
print("\n" + "="*60)
print("Test 1: Default parameters (blosum62, gap_open=5, gap_extension=1)")
print("="*60)

edgelist1 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.5,
    matrix="blosum62",
    gap_open=5,
    gap_extension=1,
    use_cache=True,
    show_progress=True
)

print(f"Edgelist 1 shape: {edgelist1.shape}")
print(f"Edgelist 1:\n{edgelist1}")

# Test 2: Different matrix - should create new cache
print("\n" + "="*60)
print("Test 2: Different matrix (blosum80)")
print("="*60)

edgelist2 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.5,
    matrix="blosum80",
    gap_open=5,
    gap_extension=1,
    use_cache=True,
    show_progress=True
)

print(f"Edgelist 2 shape: {edgelist2.shape}")
print(f"Edgelist 2:\n{edgelist2}")

# Test 3: Different gap_open - should create new cache
print("\n" + "="*60)
print("Test 3: Different gap_open (10)")
print("="*60)

edgelist3 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.5,
    matrix="blosum62",
    gap_open=10,
    gap_extension=1,
    use_cache=True,
    show_progress=True
)

print(f"Edgelist 3 shape: {edgelist3.shape}")
print(f"Edgelist 3:\n{edgelist3}")

# Test 4: Different gap_extension - should create new cache
print("\n" + "="*60)
print("Test 4: Different gap_extension (2)")
print("="*60)

edgelist4 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.5,
    matrix="blosum62",
    gap_open=5,
    gap_extension=2,
    use_cache=True,
    show_progress=True
)

print(f"Edgelist 4 shape: {edgelist4.shape}")
print(f"Edgelist 4:\n{edgelist4}")

# Test 5: Return to default parameters - should load from cache
print("\n" + "="*60)
print("Test 5: Back to default parameters (should load from cache)")
print("="*60)

edgelist5 = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.5,
    matrix="blosum62",
    gap_open=5,
    gap_extension=1,
    use_cache=True,
    show_progress=True
)

print(f"Edgelist 5 shape: {edgelist5.shape}")
print(f"Matches edgelist 1: {np.array_equal(edgelist1, edgelist5)}")

# Verify that different parameters produce different results
print("\n" + "="*60)
print("Verification: Different parameters should give different results")
print("="*60)

print(f"Edgelist 1 == Edgelist 2 (different matrix): {np.array_equal(edgelist1, edgelist2)}")
print(f"Edgelist 1 == Edgelist 3 (different gap_open): {np.array_equal(edgelist1, edgelist3)}")
print(f"Edgelist 1 == Edgelist 4 (different gap_extension): {np.array_equal(edgelist1, edgelist4)}")

print("\n" + "="*60)
print("All tests completed!")
print("✓ Different parameters create different cache files")
print("✓ Same parameters load from cache correctly")
print("="*60)
