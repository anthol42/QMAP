# Deeplearning based model to approximate identity calculation between two peptides

## TODO
- [ ] Learned pooling to produce the final weight representation
- [ ] Use multiple layers to produce the final embedding
- [ ] Regularization
  - [ ] Embedding diversity: Add a loss term to encourage the embeddings to spread in the full space (Example: cosine-based between embeddings in batch or variance feature wise - maximize var)
  - [ ] Orthogonality regularization: Add a loss term that encourages the model to produce uncorrelated features based on the covariance matrix
  - [ ] Space smoothness: Add a loss term to ensure that the space is smooth - meaning that two similar sequence must have similar embeddings: Ensure that the l2 distance between embeddings is proportional to the identity between the sequences
- [ ] Two stage training (MLM, then fine-tuning)
- [ ] Hard Negative Mining (Train a model to predict error and generate annotations for these)

## Things to look into
- [ ] Text and Code Embeddings by Contrastive Pre-Training
- [ ] Sentence-BERT
- [ ] CLIP
- [ ] Losses
  - [ ] InfoNCE
  - [ ] NT-Xent
  - [ ] Triplet loss