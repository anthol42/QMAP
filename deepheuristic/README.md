# Deeplearning based model to approximate identity calculation between two peptides

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
- [ ] Different encoders (ProtT5, ProtBERT, etc)
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