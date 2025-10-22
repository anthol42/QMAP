import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def diversity_loss(embeddings):
    """
    Compute the diversity loss for a batch of embeddings. In other words, at what point embeddings are similar
    within the batch.
    """
    # embeddings: [batch_size, embed_dim]
    normalized_emb = F.normalize(embeddings, dim=1)
    similarity_matrix = torch.mm(normalized_emb, normalized_emb.t())

    # We want low similarity between different sequences
    # Remove diagonal (self-similarity = 1)
    mask = torch.eye(similarity_matrix.size(0), device=embeddings.device).bool()
    off_diagonal = similarity_matrix.masked_fill(mask, 0)

    # Penalize high similarities
    diversity_loss = torch.mean(torch.abs(off_diagonal))
    return diversity_loss

def variance_diversity_loss(embeddings):
    """
    Compute the variance diversity loss for a batch of embeddings along features.
    """
    # Encourage high variance across embedding dimensions
    dim_variances = torch.var(embeddings, dim=0)
    return -torch.mean(dim_variances)


def orthogonality_loss(embeddings):
    """
    Compute the orthogonality loss for a batch of embeddings along features. It forces the model to output orthogonal
    features dimensions.
    """
    # embeddings: [batch_size, embed_dim]
    # Compute covariance matrix across the batch
    centered = embeddings - torch.mean(embeddings, dim=0)
    cov_matrix = torch.mm(centered.t(), centered) / (embeddings.size(0) - 1)

    # Penalize off-diagonal elements (correlations between dimensions)
    off_diagonal_mask = ~torch.eye(cov_matrix.size(0)).bool()
    off_diagonal_loss = torch.mean(torch.abs(cov_matrix[off_diagonal_mask]))

    return off_diagonal_loss
