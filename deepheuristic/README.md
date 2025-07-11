# Deeplearning based model to approximate identity calculation between two peptides
## Results
### Phase 1
In this phase, I tested multiple different configurations (79 to be exact) and did a hyperparameter search to find the 
best configuration. I used a model to make an embedding for a sequence. The embeding vector is normalized such that its 
l2 norm is 1. Then, the identity between two sequences is computed by taking the vector product between the two 
embeddings. Note that this can range between -1 and 1 whereas identity is between 0 and 1. So, I used a Parametric ReLU 
to transform the scores to the [0, 1] range. The model then learned the optimal parameters of the PReLU.

What I found is that the AdamW optimizer outperform the Adam optimizer with some weight decay because the training is 
unstable with Adam. However, after a hyperparameter search, I found that training without weight decay and using early 
stopping is better. After, I notice a loglinear improvement with the embedding size with a linear projection layer after 
the backbone. I also noticed an improvement with the backbone size.

I noticed that using a mean over the sequence embeddings is better than using the CLF token. This is probably because it 
is less sensitive to extreme samples.

After, I noticed that using an MLP head instead of a linear projection does not improve the performances, which is 
weird. This is why I did a two branch MLP head with a linear head and a MLP head. This improved the performances. Then, 
I tested multiple normalization between layers (ESM, layer norm, batch norm, no norm) and found that the batch norm is 
the best one, but it does not outperform the no norm. Also, I noticed that when using normalization (which worsen the 
results), using pre-norm was better than post-norm. I also noticed that using skip connections in the MLP branch 
didn't improved training performances, but improved generalization performances.

After, I wondered if using a learned pooling instead of a mean pooling would further improve the performances. I tested 
it and found that it does not help. However, using a learned pooling across layers does increase generalization. This 
works by summing the mean embeddings of each layers with a learned weight vector. The learned weight vector is 
initialized to all ones. 

I also noticed that using an EMA on the backbone weights considerably improves the generalization performances.

I tried using an MLP to predict the identity score instead of doing a vector product between the normalized embeddings, 
but no matter what I tried, it did not improve the performances. The main thing to think of is that the identity between 
two sequences is indepentent of the order of the sequences. This means that simply concatenating the two embeddings 
won't work whereas the vector product is invariant to the order of the sequences. I tried doing an absolute difference 
between the embeddings, a element wise product and a concatenation, but none of these worked. The model either overfit or
did not converge well.

Finally, I tried different regularization techniques to improve the embedding space. I tried the diversity metric, 
which encourages the embeddings generated in each batch to be as diverse as possible (their dot product is as close to 0 
as possible). I also tried the orthogonality regularization, which encourages the embeddings features to be as 
uncorrelated (a diagonal covariance matrix in each batch). I also tried the space smoothness regularization, 
which encourages the embeddings of similar sequences to be close in l2 norm. This encourages the space to be aligned in 
terms of angular distances and l2 distances (Correlation between the two). I also tried a variance regularization which 
is similar to the diversity regularization, but it encourages the variance of each feature to be as high as possible 
within each batch. I noticed that all of these regularization techniques improve the generalization performances, but 
the tuning of the weight of each is really important and dependent on the technique. This means that we can't use the 
same weight for all. It would be interesting to do a hyperparameter search to learn these weights. After combining all 
of these regularization techniques, I notices that it was better to use three of the four regularization techniques: 
diversity, orthogonality and smoothness. The variance regularization did not combined well with the others and 
did not produce better results (Note that using the var regularization increase the performances in test (best perf), 
but not in val, so I consider this luck and not the best run. [RunID 79]). 

After all these tests, I noticed that the performances on the DBAASP dataset are way worse than on the test set, which 
is odd because the test set is supposed to represent the real world performances. I double check the code and did not 
find any typo or mistake. Then, I did a PCA projection of the test embeddings and the DBAASP embeddings and found that
the DBAASP embeddings seems to be within the distribution of the test embeddings. This did not help uncover the 
difference. Then, I trained an LDA model to try to discriminate between the test embeddings and the DBAASP ones. The 
base hypothesis is that the balanced accuracy will be close to 50% since the distribution should be similar. However, 
I obtained 84% balanced accuracy, which is really high. This means that the two distributions are not similar and that 
a linear model can discriminate between the two. As a control, I also tried this experiment with the validation set. 
The LDA model was  able to discriminate between the validation set and the test set with a balanced accuracy of 54%, 
which is way lower, and as expected. This means that the DBAASP dataset is really out of distribution, but why? 

By looking at the good predictions of the LDA model, I noticed that most sequences have a repeated pattern or contains 
only few different amino acids. I dubbed those sequence 'low complexity sequences'. I implemented a shannon entropy 
measure function to compute the the complexity of the sequences. I found that the Peptide atlas dataset has a mean 
complexity of 3.5 while the DBAASP dataset has a mean complexity of 2.5. By looking at the distributions, I found that 
the complexity distribution of the DBAASP dataset is a lot wider than the Peptide atlas dataset with a heavy tail 
towards low complexity sequences. 

To verify this hypothesis, I computed the performances of the model on low complexity sequences and found similar 
performances as the one obtained on the DBAASP dataset. 

To improve the performances for AMPs, we will need to improve the dataset by adding low complexity sequences. The 
phase 2 of the project will use the new low-complexity dataset to train and evaluate the model

## TODO
- [X] Different norm and no norm in MLP head
- [X] Try pre- and post-norm
- [X] Learned pooling to produce the final weight representation
- [X] Use multiple layers to produce the final embedding
- [X] EMA
- [X] Regularization
  - [X] Embedding diversity: Add a loss term to encourage the embeddings to spread in the full space (Example: cosine-based between embeddings in batch or variance feature wise - maximize var)
  - [X] Orthogonality regularization: Add a loss term that encourages the model to produce uncorrelated features based on the covariance matrix
  - [X] Space smoothness: Add a loss term to ensure that the space is smooth - meaning that two similar sequence must have similar embeddings: Ensure that the l2 distance between embeddings is proportional to the identity between the sequences
- [ ] Different encoders
  - [ ] ESM2 8M, 35M, 150M, 650M, 3B
  - [ ] Ankh (Base (1.5B, Large 3B)
  - [ ] ProstT5 (5B)
  - [ ] Prot_t5_xl_uniref50 (3B)
- [ ] Two stage training (Pre training, then fine-tuning)
  - [ ] MLM
  - [ ] Contrastive
  - [ ] Self distillation
  - [ ] MLM + Contrastive
  - [ ] MLM + Self distillation
- [ ] Hard Negative Mining (Train a model to predict error and generate annotations for these)
- [ ] Refiner (Estimate the uncertainty using the vector approach (like current), when the uncertainty is over a 
threshold, we refine the identity using a more powerful, but slower model)

## Things to look into
- [ ] Text and Code Embeddings by Contrastive Pre-Training
- [ ] Sentence-BERT
- [ ] CLIP
- [ ] Losses
  - [ ] InfoNCE
  - [ ] NT-Xent
  - [ ] Triplet loss