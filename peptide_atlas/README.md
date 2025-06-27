# Peptide Atlas dataset
This document explains how to prepare the Peptide Atlas dataset and get it ready to train the model.

# Download the dataset
1. Sync the repository using uv:
```bash
uv sync
```
2. Download the peptide atlas database by running the `download.py` scipt:
```bash
uv run download.py
```
3. Split the dataset into train, validation, and test sets by running the `split_dataset.py` script:
```bash
uv run split_dataset.py
```
4. The dataset splits will be saved in the build directory. Next, you need to annotate the dataset. This task is 
compute intensive and may take a while to complete. To make things faster, we suggest to split the computation across 
multiple processes or even compute nodes. To build the original dataset, the following process were run:
- Train: 100 processes of 2.5M alignments each
- Validation: 15 processes of 2.5M alignments each
- Test: 30 processes of 2.5M alignments each

An example to run the annotation for a single process for the training set is:
```bash
uv run make_alignments.py --input=build/train.fasta --output=.cache/train_parts/0.npy
```
Note that the sequences to annotate (align) are sampled randomly. Since the combination space is much larger than the 
sampled alignments, we do not need to check if the alignments are unique. Thus, each process can run independently and 
the results can be merged later.
5. After all processes are done, you can merge the results by running the `merge_alignments.py` script:
```bash
uv run merge_alignments.py --dir=.cache/val_parts --output=build/val.npy
uv run merge_alignments.py --dir=.cache/test_parts --output=build/test.npy
uv run merge_alignments.py --dir=.cache/train_parts --output=build/train.npy
```

Done! The dataset is now ready!
