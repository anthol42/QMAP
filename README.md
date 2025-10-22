# QMAP benchmark

Here is the code to reproduce the results and figures of our paper. There is also the source code of the `qmap-benchmark` PyPi package in the `QMAP` directory.


## Install QMAP-benchmark
```shell
pip install qmap-benchmark
```

## Package documentation
The documentation formatted as markdown is available in [QMAP/docs/references](QMAP/docs/references)  
Examples are shown in [QMAP/docs/examples](QMAP/docs/examples)

## Reproduce the results
To reproduce the results of the paper, you must first fetch and prepare the [data](data/README.md).

Then, you can run what you are interested in:)

The repo is structured as follow:
- [`data`](data/README.md): Code to download and prepare the data used in the project.
- [`data_leakage`](data_leakage/README.md): Code to quantitatively evaluate the data leakage induced by different methods. This code also generates the maximum identity distribution figures from the paper.
- [`eval_prev_works`](eval_prev_works/README.md): Code to evaluate previous methods on the QMAP benchmark and to draw the figures showing the performances.
- [`figures`](figures/README.md): Contains multiple notebooks to generate additional visualizations.
- [`deep`](deep/README.md): Contains the code to train the deep learning-based encoder that serves to approximate the pairwise identity matrix.

- [`QMAP`](QMAP/README.md): The PyPi package code.

## Cite
If you use our work, please cite us

```
TODO!!!
```