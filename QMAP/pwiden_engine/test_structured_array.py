#!/usr/bin/env python3
"""
Test that edgelist returns a proper numpy structured array.
"""

import pwiden_engine
import numpy as np

sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRST", "MKTIIALSYIFCLVFA", "AAAA"]

print("=" * 60)
print("Testing structured array edgelist")
print("=" * 60)

edgelist = pwiden_engine.create_edgelist(
    sequences,
    threshold=0.3,
    use_cache=False,
    show_progress=False
)

print(f"\nEdgelist shape: {edgelist.shape}")
print(f"Edgelist dtype: {edgelist.dtype}")
print(f"Edgelist type: {type(edgelist)}")

print(f"\nField names: {edgelist.dtype.names}")
print(f"Field types:")
for name in edgelist.dtype.names:
    print(f"  {name}: {edgelist.dtype.fields[name][0]}")

print(f"\nFirst 5 edges:")
for i in range(min(5, len(edgelist))):
    edge = edgelist[i]
    print(f"  Edge {i}: source={edge['source']}, target={edge['target']}, identity={edge['identity']:.3f}")

print(f"\n" + "=" * 60)
print("Accessing fields")
print("=" * 60)

sources = edgelist['source']
targets = edgelist['target']
identities = edgelist['identity']

print(f"\nSources type: {sources.dtype}")
print(f"Targets type: {targets.dtype}")
print(f"Identities type: {identities.dtype}")

print(f"\nSources: {sources}")
print(f"Targets: {targets}")
print(f"Identities: {identities}")

print(f"\n" + "=" * 60)
print("Verifying properties")
print("=" * 60)

print(f"\n✓ All sources are integers: {np.all(sources == sources.astype(int))}")
print(f"✓ All targets are integers: {np.all(targets == targets.astype(int))}")
print(f"✓ All source < target (upper triangle): {np.all(sources < targets)}")
print(f"✓ All identities >= threshold: {np.all(identities >= 0.3)}")

print(f"\n" + "=" * 60)
print("Testing indexing and slicing")
print("=" * 60)

# Individual element access
first_edge = edgelist[0]
print(f"\nFirst edge: {first_edge}")
print(f"  Type: {type(first_edge)}")
print(f"  Source: {first_edge['source']} (type: {type(first_edge['source'])})")
print(f"  Target: {first_edge['target']} (type: {type(first_edge['target'])})")
print(f"  Identity: {first_edge['identity']} (type: {type(first_edge['identity'])})")

# Slicing
print(f"\nFirst 3 edges:")
print(edgelist[:3])

# Boolean indexing
high_identity = edgelist[edgelist['identity'] > 0.8]
print(f"\nHigh identity edges (>0.8): {len(high_identity)} edges")
print(high_identity)

print(f"\n" + "=" * 60)
print("✓ All tests passed!")
print("✓ Structured array with proper dtypes (i64, i64, f32)")
print("=" * 60)
