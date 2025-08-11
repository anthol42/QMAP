# Antimicrobial-Peptides
This directory contains a modified version of [Antimicrobial-Peptides](https://github.com/zswitten/Antimicrobial-Peptides)

It was modified only to use it with the QMAP benchmark. In addition, code that was not used to train the model was removed.

## Running
The new parameter is `option`. You can choose between `original` and `qmap`.
- `original`: Run the code with their random train test split.
- `qmap`: Run the code, but with the independent qmap benchmark.

```shell
uv run main.py --negatives=1 --bacterium='E. coli' --epochs=60 --option='qmap'
```

## Results
For e. coli

| Split | Threshold | RMSE | MSE | MAE | R2 | Spearman | Kendall's Tau | Pearson |
|-------|-----------|------|-----|-----|----|----------|----------------|---------|
| 0 | 55 | 0.9873 | 0.9749 | 0.7817 | 0.2252 | 0.4870 | 0.3377 | 0.4900 |
| 1 | 55 | 0.9567 | 0.9152 | 0.7564 | 0.2233 | 0.5138 | 0.3579 | 0.5106 |
| 2 | 55 | 0.9453 | 0.8935 | 0.7424 | 0.1944 | 0.4406 | 0.3039 | 0.4757 |
| 3 | 55 | 0.9491 | 0.9008 | 0.7626 | 0.2113 | 0.4640 | 0.3168 | 0.4771 |
| 4 | 55 | 1.0013 | 1.0026 | 0.8074 | 0.1596 | 0.4099 | 0.2854 | 0.4408 |
| 0 | 60 | 0.9354 | 0.8750 | 0.7267 | 0.2153 | 0.4461 | 0.3104 | 0.4640 |
| 1 | 60 | 1.0110 | 1.0221 | 0.7968 | 0.1147 | 0.3327 | 0.2241 | 0.3524 |
| 2 | 60 | 1.0848 | 1.1767 | 0.8631 | 0.1661 | 0.3923 | 0.2687 | 0.4097 |
| 3 | 60 | 0.9881 | 0.9763 | 0.7726 | 0.0650 | 0.2967 | 0.2019 | 0.3210 |
| 4 | 60 | 0.9861 | 0.9723 | 0.7892 | 0.0579 | 0.3331 | 0.2285 | 0.3608 |