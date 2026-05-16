# Evaluation of previous works.

We downloaded the code of several previous works that predict the MIC of peptides against bacterial targets. Those 
downloaded code were slightly modified in order to evaluate them on the QMAP 
benchmark. Each directory contains the code of a previous work. The code trains the model, then 
evaluates it on the QMAP benchmark for the five splits. The results are saved in the `results` folder inside each 
directory.

In addition, two baseline linear probing model are trained on the ESM650M embeddings, one for MIC and the other for 
HC50. They are implemented in the `Linear` and `HemoLinear` directories respectively.

The three notebooks generates the figures presented in the manuscript.

To rerun the experiments on the benchmark, simply run each main file in each directory. Make sure to activate the 
right uv environment for each previous works as they do not have the same python versions and dependencies.

## Reproduce results

All commands are run from within each project directory. Each project has its own `uv` environment.

### Linear

```shell
cd Linear/
uv run main.py                   # full.csv, high_efficiency.csv
uv run main.py --train_he        # train_he.csv
uv run main.py --train_me        # train_me.csv
uv run main.py --train_le        # train_le.csv
uv run main.py --rnd             # rnd_split.csv
```

### Antimicrobial-Peptides

```shell
cd Antimicrobial-Peptides/
uv run main.py                   # full.csv, high_efficiency.csv
uv run main.py --train_he        # train_he.csv
uv run main.py --train_me        # train_me.csv
uv run main.py --train_le        # train_le.csv
uv run main.py --rnd             # rnd_split.csv
```

### AMP_regression_EC_SA

```shell
cd AMP_regression_EC_SA/
uv run main.py                              # full.csv, high_efficiency.csv
uv run main.py --earlystoping               # full_es.csv, high_efficiency_es.csv
uv run main.py --train_he                   # train_he.csv
uv run main.py --train_me                   # train_me.csv
uv run main.py --train_le                   # train_le.csv
uv run main.py --train_he --earlystoping    # train_he_es.csv
uv run main.py --train_me --earlystoping    # train_me_es.csv
uv run main.py --train_le --earlystoping    # train_le_es.csv
uv run main.py --rnd                        # rnd_split.csv
```
