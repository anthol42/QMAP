# QMAP-Benchmark Project Guide

## Overview
QMAP-Benchmark is a Python toolkit for benchmarking antimicrobial peptide (AMP) MIC regression and hemolytic regression. It provides utilities to split peptide sequence datasets into independent sets based on global identity.

## Project Structure

```
src/qmap/
├── benchmark/          # Benchmark datasets and metrics
│   ├── dataset/       # Dataset implementations (bond, hemolytic, etc.)
│   ├── metrics.py     # Evaluation metrics
│   └── benchmark.py   # Main benchmark class
└── toolkit/           # Data processing utilities
    ├── split/         # Train/test splitting strategies
    ├── aligner/       # Python wrapper of the rust pwiden_engine for fast pairwise identity calculation.
    └── clustering/    # Sequence clustering tools
```

## Key Components

- **Datasets**: Bond and hemolytic datasets with filters and sampling
- **Splitting**: Identity-based train/test splitting to avoid data leakage
- **Clustering**: Graph-based clustering using igraph and Leiden algorithm
- **Metrics**: Custom QMAP metrics for evaluation

## Dependencies
- PyTorch for deep learning
- igraph + leidenalg for clustering
- pandas/numpy for data manipulation
- huggingface-hub for dataset distribution

## Important Notes
- Python ≥3.9 required
- Uses Rust extensions (pwiden_engine/) for identity computation
- Focuses on preventing data leakage through sequence similarity
