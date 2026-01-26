"""
Test script to verify that the QMAP package installation was successful.
Tests imports from both pwiden_engine (Rust extension) and qmap (Python package).

Run with: pytest test_installation.py -v
"""

import pytest


class TestPwidenEngineImports:
    """Test imports from the pwiden_engine Rust extension."""

    def test_import_compute_binary_mask(self):
        from pwiden_engine import compute_binary_mask

    def test_import_compute_global_identity(self):
        from pwiden_engine import compute_global_identity

    def test_import_compute_maximum_identity(self):
        from pwiden_engine import compute_maximum_identity

    def test_import_create_edgelist(self):
        from pwiden_engine import create_edgelist

    def test_import_get_cache_dir(self):
        from pwiden_engine import get_cache_dir


class TestQmapImports:
    """Test imports from the qmap Python package."""

    def test_import_dbaaspdataset(self):
        from qmap import DBAASPDataset

    def test_import_identity(self):
        from qmap import Identity

    def test_import_qmapbenchmark(self):
        from qmap import QMAPBenchmark

    def test_import_train_test_split(self):
        from qmap import train_test_split
