# QMAP benchmark and toolkit
A domain-specific homology aware benchmark that ensure robust performance evaluation and comparison between models predicting MIC and/or HC50 (regression).

Features:
- Regression on consensus bacterial MIC and mammal HC50
- Predefined splits, to ensure comparability
- N-terminal acetylation and C-terminal amidation
- Common noncannonical amino acids O: Ornithine and B: 2,4-Diaminobutyric acid
- Non-cannonical peptides as SMILES
- Intrachain bonds: 
- Machine learning ready dataset
- Extendable training dataset, so you can use your own data.

## Install QMAP-benchmark
```shell
pip install qmap-benchmark
```

## Get started
```shell
# Import the DBAASP training dataset and the benchmark
from qmap import QMAPBenchmark, DBAASPDataset, train_test_split

dataset = DBAASPDataset()
# We need to test on each split, and train on an independent set for each split.
for split in range(5):
    benchmark = QMAPBenchmark(split=split)

    train_mask = benchmark.get_train_mask(dataset.sequences)
    train_dataset = dataset[train_mask]
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.15, random_state=42)
    
    # Train on train_dataset ...
    ...
    # Predict the benchmark dataset
    preds = ...
    
    # The benchmark automatically computes the metrics!
    test_perf = benchmark.compute_metrics(preds)
    print(test_perf)
```

## Leaderboard
> **Add an entry:**
>
> Here is the leaderboard. Please open an issue to add your model to the leaderboard with the code to reproduce the results, and the reference paper.

### Full - e. coli
| Method | Year | e. coli min PCC | e. coli mean | e. coli max PCC | Source Code |
|--------|------|-----------------|--------------|-----------------|--------|
|Linear model on ESM2 embeddings| N/A | 0.32. | 0.36 |  0.41 | [`eval_prev_works/Linear`](eval_prev_works/Linear)|
| [J. Witten and Z. Witten](https://doi.org/10.1101/692681) | 2019 | 0.47 | 0.51 | 0.56 | [`eval_prev_works/Antimicrobial-Peptides`](eval_prev_works/Antimicrobial-Peptides)
| [J. Cai et al](https://doi.org/10.1021/acs.jcim.4c01749)| 2025 | 0.47 | 0.52 | 0.56 | [`eval_prev_works/AMP_regression_EC_SA`](eval_prev_works/AMP_regression_EC_SA)|

### High efficiency - e. coli
| Method | Year | e. coli min PCC | e. coli mean | e. coli max PCC | Source Code |
|--------|------|-----------------|--------------|-----------------|--------|
|Linear model on ESM2 embeddings| N/A | 0.06 | 0.16 |  0.22 | [`eval_prev_works/Linear`](eval_prev_works/Linear)|
| [J. Witten and Z. Witten](https://doi.org/10.1101/692681) | 2019 | 0.10 | 0.22 | 0.33 | [`eval_prev_works/Antimicrobial-Peptides`](eval_prev_works/Antimicrobial-Peptides)
| [J. Cai et al](https://doi.org/10.1021/acs.jcim.4c01749)| 2025 | 0.20 | 0.29 | 0.33 | [`eval_prev_works/AMP_regression_EC_SA`](eval_prev_works/AMP_regression_EC_SA)|

### Full - hc50
| Method | Year | e. coli min PCC | e. coli mean | e. coli max PCC | Source Code |
|--------|------|-----------------|--------------|-----------------|--------|
|Linear model on ESM2 embeddings| N/A | -0.18 | 0.07 |  0.29 | [`eval_prev_works/HemoLinear`](eval_prev_works/HemoLinear)|

## Documentation
The documentation formatted as markdown is available in [QMAP/docs/references](https://github.com/anthol42/QMAP/tree/main/QMAP/docs/references)  
Examples are shown in [QMAP/docs/examples](https://github.com/anthol42/QMAP/tree/main/QMAP/docs/examples)

## Please Cite
Please cite us if you find yourself using our work

```
@misc{lavertu_qmap_2026,
    title = {{QMAP}: {A} {Benchmark} for {Standardized} {Evaluation} of {Antimicrobial} {Peptide} {MIC} and {Hemolytic} {Activity} {Regression}},
    url = {https://www.biorxiv.org/content/10.64898/2026.02.03.703041v1},
    doi = {10.64898/2026.02.03.703041},
    publisher = {bioRxiv},
    author = {Lavertu, Anthony and Corbeil, Jacques and Germain, Pascal},
    month = feb,
    year = {2026}
}
```