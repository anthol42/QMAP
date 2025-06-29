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
3. Cluster the sequences using CD-HIT by running the `cluster_cdhit.sh` script:
```bash
./cluster_cdhit.sh
```
4. Split the dataset into train, validation, and test sets by running the `split_dataset.py` script:
```bash
uv run split_dataset.py
```
5. Next, you need to create the clusters for each split.:
```shell
cd-hit -i build/train.fasta -o .cache/train_clusters -c 0.8 -n 2 -d 0 -M 30000 -T 10 -l 5
cd-hit -i build/val.fasta -o .cache/val_clusters -c 0.5 -n 2 -d 0 -M 30000 -T 10 -l 5
cd-hit -i build/test.fasta -o .cache/test_clusters -c 0.5 -n 2 -d 0 -M 30000 -T 10 -l 5
```
6. The dataset splits will be saved in the build directory. Next, you need to annotate the dataset. This task is 
compute intensive and may take a while to complete. To make things faster, we suggest to split the computation across 
multiple processes or even compute nodes. To build the original dataset, the following process were run:
- Train: 100 processes of 2.5M alignments each with random sampling and 100 processes of 1.25M alignments with cluster sampling
- Validation: 15 processes of 2.5M alignments each with random sampling and 15 processes of 2.5M alignments with cluster sampling
- Test: 30 processes of 2.5M alignments each with random sampling and 30 processes of 2.5M alignments with cluster sampling

An example to run the annotation for a single process for the training set is:
```bash
uv run make_alignments.py --input=build/train.fasta --output=.cache/train_parts/0.npy --type=random
uv run make_alignments.py --input=build/train.fasta --output=.cache/train_parts/100.npy --type=cluster
```
Note that the sequences to annotate (align) are sampled randomly for the random sampling. Since the combination space 
is much larger than the sampled alignments, we do not need to check if the alignments are unique. Thus, each process 
can run independently and the results can be merged later. The same is true for the cluster sampling. We need to do 
both types of sampling to get an identity distribution close to uniform.
7. After all processes are done, you can merge the results by running the `merge_alignments.py` script:
```bash
uv run merge_alignments.py --dir=.cache/val_parts --output=build/val.npy --min_samples=10000
uv run merge_alignments.py --dir=.cache/test_parts --output=build/test.npy --min_samples=30000
uv run merge_alignments.py --dir=.cache/train_parts --output=build/train.npy --min_samples=100000
```

Done! The dataset is now ready!
