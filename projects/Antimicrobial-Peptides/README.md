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
### RMSE
| Threshold | 0    | 1     | 2    | 3    | 4    |
|-----------|------|-------|------|------|------|
| 0.55      | 0.98 | 0.95  | 0.90 | 0.92 | 0.99 |
| 0.60      |      |       |      |      |      |

### PCC
| Threshold | 0    | 1    | 2    | 3    | 4    |
|-----------|------|------|------|------|------|
| 0.55      | 0.50 | 0.53 | 0.53 | 0.52 | 0.51 |
| 0.60      |      |      |      |      |      |

### Kendall tau
| Threshold | 0    | 1    | 2    | 3    | 4    |
|-----------|------|------|------|------|------|
| 0.55      | 0.34 | 0.38 | 0.35 | 0.35 | 0.33 |
| 0.60      |      |      |      |      |      |

### MSE
| Threshold | 0    | 1    | 2    | 3    | 4    |
|-----------|------|------|------|------|------|
| 0.55      | 0.97 | 0.90 | 0.82 | 0.85 | 0.98 |
| 0.60      |      |      |      |      |      |

### MAE
| Threshold | 0   | 1   | 2   | 3   | 4   |
|-----------| --- | --- | --- | --- | --- |
| 0.55      | | | | | |
| 0.60      | | | | | |
