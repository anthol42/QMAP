# Split experiments
In this experiment, we want to quantitatively measure the amount of data leakage induced by different dataset splitting 
methods. We evaluate four different methods, one of which is our novel method. 

## Get started
### 1. Download the data
First, you need to fetch and build the DBAASP dataset. You can do so by following the instructions in the 
`/data/README.md` directory.

### 2. Create the ground truth identity matrix
Next, you must first create the global pairwise identity matrix that will be used to evaluate the splitting 
performances of each method. It is highly recommended to parallelize this step, as it is very costly in terms of 
compute and the algorithm is single-threaded. To go so, you can run the `ground_truth/compute_identity` script with the
range of sequences to compute in this call. For example, if you want to use 100 cores:
```shell
uv run compute_identity.py --input=../data/build/dataset.fasta --id_range=0-200
uv run compute_identity.py --input=../data/build/dataset.fasta --id_range=200-400
...
uv run compute_identity.py --input=../data/build/dataset.fasta --id_range=19600-19800
uv run compute_identity.py --input=../data/build/dataset.fasta --id_range=19800-19038
```
Do this for all sequences up to 19038.

Once all script are run, you can assemble the global identity matrix and move it to the .cache directory by running:
```shell
uv run assemble_matrix.py --input=matrix_parts
```

### 3. Install dependencies

Now, you need to install mmseqs2 and cd-hit. On mac, I installed mmseqs2 using bio-conda and cd-hit by building it from 
source.

Then, you can run the notebooks in the different directories to explore the different methods.