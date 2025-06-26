# Split experiments
In this project, we will explore different methods to split the dataset into training and testing sets. We notice that
not all methods are equally robust. The goal is to have a certain level of control on how test sequences must be 
different from training sequences to ensure that we accurately evaluate the model's generalization performance.

To evaluate this, we will compute certain statistics on the identity between the sequences of the training and testing 
sets for different splitting methods.

## Methods
### Random Split
We will shuffle the sequences, then naively split them.

### CD-Hit
This method has been used extensively by multiple older papers. It clusters the sequences according to a certain 
similarity threshold. Then, we will select multiple clusters as the train set and the rest as the test set.

### MMseqs2
This method is more recent than CD-Hit and is supposedly more efficient. It also clusters the sequences according to a 
certain similarity threshold. The, the same method will be used to select the train and test sets.


### Community detection (Leiden algorithm)
For this experiment, we assumes we are able to obtain the global identity matrix between all sequences. Then, we can 
build a graph by connecting the sequences with edges if their identity is above a certain threshold. Then, we can split 
the dataset by applying a community detection algorithm, such as the Leiden algorithm, to find communities in the graph.
Those communities can be understood as clusters of sequences as the previous methods. We can then select some 
communities as the train set and the rest as the test set.

This method is more theoretical than practical because it requires the global identity matrix, which is really costly 
in terms of compute. This tells us: if I am ready to compute the global identity matrix, will the community detection 
algorithm be better than the clustering methods?

### Deep learning heuristic
CD-Hit and MMSeqs2 use heuristics to split the sequences by identity because it would be too expensive to compute the 
global identity matrix using dynamic programming. In this method, we will implement a deep learning model that will 
learn to predict the identity between two sequences. We will then use this model to compute the global identity matrix
between all sequences. 

Then, we will build a graph from it and apply the community detection algorithm to split the dataset.

## Get started
First, you need to fetch and build the DBAASP dataset. You can do so by following the 
`Download and build the FASTA dataset` instructions in the `data/README.md` file.

Next, you must first create the global identity matrix that will be used to evaluate the splitting 
performances of each method. It is highly recommended to parallelize this step, as it is very costly in terms of 
compute and the algorithm is single-threaded. To go so, you can run the *******TODO*********
## Results
