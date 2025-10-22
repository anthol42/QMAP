import torch
import torch.nn.functional as F
from torch import nn

class LinearActivation(nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return 2*X - 1

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


class Criterion(nn.Module):
    def __init__(self, diversity: float = 0., var: float = 0.,
                 orthogonality: float = 0., smoothness: float = 0.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.activation = nn.PReLU(init=0.25)

        self.diversity = diversity
        self.var = var
        self.orthogonality = orthogonality
        self.smoothness = smoothness

    def forward(self, pred1, pred2, target):
        pred = (pred1.unsqueeze(1) @ pred2.unsqueeze(2)).squeeze(-1)
        # Normalize between 0 and 1 (Soft with PReLU)
        pred = self.activation(pred)
        loss = self.criterion(pred, target)
        embs = torch.cat([pred1, pred2], dim=0)
        if self.smoothness > 0:
            l2 = torch.norm(pred1 - pred2, p=2, dim=1)
            added = self.smoothness * ((l2 - target)**2).mean()
            loss += added
        if self.diversity > 0:
            added = self.diversity * diversity_loss(embs)
            loss += added
        if self.var > 0:
            added = self.var * 5 * variance_diversity_loss(embs) # Multiply by 5 to be in the same order as the loss
            loss += added
        if self.orthogonality > 0:
            added = self.orthogonality * 100 * orthogonality_loss(embs) # Multiply by 100 to be in the same order as the loss
            loss += added
        return loss, pred