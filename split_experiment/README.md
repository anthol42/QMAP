# Split experiments
In this project, we will explore different methods to split the dataset into training and testing sets. We notice that
not all methods are equally robust. The goal is to have a certain level of control on how test sequences must be 
different from training sequences to ensure that we accurately evaluate the model's generalization performance.

To evaluate this, we will compute certain statistics on the identity between the sequences of the training and testing 
sets for different splitting methods.

## Methods
### Random Split
We will shuffle the sequences, then naively split them.

## CD-Hit
This method has been used extensively by multiple older papers. It clusters the sequences according to a certain 
similarity threshold. Then, we will select multiple clusters as the train set and the rest as the test set.

## MMseqs2
This method is more recent than CD-Hit and is supposedly more efficient. It also clusters the sequences according to a 
certain similarity threshold. To ensure that we have a fair comparison, we will use parameters that are said to be 
equivalent to CD-Hit by MMSeqs2 documentation. The, the same method will be used to select the train and test sets.

## Deep learning heuristic
CD-Hit and MMSeqs2 use heuristics to split the sequences by identity because it would be too expensive to compute the 
global identity matrix using dynamic programming. In this method, we will implement a deep learning model that will 
learn to predict the identity between two sequences. We will then use this model to compute the identity matrix and
split the sequences according to a certain threshold. 

From this matrix, we can apply a threshold and convert it to a 
binary matrix. This matrix can be seen as an adjacency matrix of a graph, where the nodes are the sequences and the 
edges connect similar sequences. Ideally, we would like to sample graph components as train and test sets, but this is
unlikely to happened in practice because most components might be connected by few edges. **SOLUTION?**

## Results
