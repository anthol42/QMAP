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
