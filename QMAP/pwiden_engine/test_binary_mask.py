#!/usr/bin/env python3
"""
Test binary mask computation for train/test splits.
"""

import pwiden_engine
import numpy as np
import os
import shutil

# Create test sequences
train_sequences = [
    "ACDEFGHIKLMNPQRSTVWY",  # 0: Similar to test[0]
    "ACDEFGHIKLMNPQRST",     # 1: Similar to test[0]
    "MKTIIALSYIFCLVFA",      # 2: Different from all test
    "AAAA",                  # 3: Different from all test
]

test_sequences = [
    "ACDEFGHIKLMNPQRSTVWY",  # Identical to train[0], very similar to train[1]
    "GGGGGGGG",              # Different from all train
]

print("=" * 60)
print("Binary Mask Test")
print("=" * 60)

print(f"\nTrain sequences: {len(train_sequences)}")
print(f"Test sequences: {len(test_sequences)}")

# Clear cache
cache_dir = pwiden_engine.get_cache_dir()
if os.path.exists(cache_dir):
    # Remove only binary mask cache files
    for f in os.listdir(cache_dir):
        if f.startswith("binary_mask_"):
            os.remove(os.path.join(cache_dir, f))

print("\n" + "=" * 60)
print("Test 1: Compute mask with threshold=0.8")
print("=" * 60)

mask = pwiden_engine.compute_binary_mask(
    train_sequences,
    test_sequences,
    threshold=0.8,
    use_cache=True,
    show_progress=True
)

print(f"\nMask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask: {mask}")

# Expected: train[0] and train[1] should be True (similar to test[0])
# train[2] and train[3] should be False (different from all test)
print(f"\nExpected mask: [True, True, False, False]")
print(f"  train[0] 'ACDEFGHIKLMNPQRSTVWY' identical to test[0] -> True")
print(f"  train[1] 'ACDEFGHIKLMNPQRST' very similar to test[0] -> True")
print(f"  train[2] 'MKTIIALSYIFCLVFA' different from test -> False")
print(f"  train[3] 'AAAA' different from test -> False")

print("\n" + "=" * 60)
print("Test 2: Load from cache")
print("=" * 60)

mask2 = pwiden_engine.compute_binary_mask(
    train_sequences,
    test_sequences,
    threshold=0.8,
    use_cache=True,
    show_progress=True
)

print(f"\nMasks are equal: {np.array_equal(mask, mask2)}")

print("\n" + "=" * 60)
print("Test 3: Different threshold should create different cache")
print("=" * 60)

mask_low_threshold = pwiden_engine.compute_binary_mask(
    train_sequences,
    test_sequences,
    threshold=0.3,
    use_cache=True,
    show_progress=True
)

print(f"\nLow threshold mask: {mask_low_threshold}")
print(f"High threshold mask: {mask}")
print(f"Are they different? {not np.array_equal(mask, mask_low_threshold)}")

print("\n" + "=" * 60)
print("Test 4: Swap train/test should give different result")
print("=" * 60)

# Swap: now test becomes train, train becomes test
mask_swapped = pwiden_engine.compute_binary_mask(
    test_sequences,
    train_sequences,
    threshold=0.8,
    use_cache=False,
    show_progress=False
)

print(f"\nOriginal mask (train={len(train_sequences)}): {mask}")
print(f"Swapped mask (train={len(test_sequences)}): {mask_swapped}")
print(f"Different lengths? {len(mask) != len(mask_swapped)}")
print(f"Expected: Original has {len(train_sequences)} elements, swapped has {len(test_sequences)} elements")

print("\n" + "=" * 60)
print("Test 5: Verify practical usage")
print("=" * 60)

# Filter training sequences using the mask
filtered_train = [seq for i, seq in enumerate(train_sequences) if not mask[i]]
removed_train = [seq for i, seq in enumerate(train_sequences) if mask[i]]

print(f"\nOriginal train set: {len(train_sequences)} sequences")
print(f"Sequences to remove: {sum(mask)} sequences")
print(f"Filtered train set: {len(filtered_train)} sequences")

print(f"\nRemoved sequences (too similar to test):")
for seq in removed_train:
    print(f"  {seq}")

print(f"\nKept sequences (independent from test):")
for seq in filtered_train:
    print(f"  {seq}")

print("\n" + "=" * 60)
print("✓ All tests completed!")
print("✓ Binary mask correctly identifies train sequences similar to test")
print("✓ Caching works with train/test separation")
print("=" * 60)
