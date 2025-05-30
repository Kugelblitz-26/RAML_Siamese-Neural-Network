
import torch
import torch.nn.functional as F
import random


def random_sampling(labels)
    batch_size = labels.size(0)
    negatives = []
    for i in range(batch_size):
        negs = (labels != labels[i]).nonzero(as_tuple=False).view(-1)
        negatives.append(random.choice(negs.tolist()))
    return torch.tensor(negatives, device=labels.device)


def hard_negative_sampling(embeddings, labels):
    """Select hardest negative for each anchor in the batch."""
    dist = torch.cdist(embeddings, embeddings)
    batch_size = labels.size(0)
    negatives = []
    for i in range(batch_size):
        mask = labels != labels[i]
        neg_dists = dist[i][mask]
        neg_indices = mask.nonzero(as_tuple=False).view(-1)
        hardest = torch.argmin(neg_dists)
        negatives.append(neg_indices[hardest].item())
    return torch.tensor(negatives, device=labels.device)


def semi_hard_negative_sampling(embeddings, labels, margin=1.0):
    """Select semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin."""
    dist = torch.cdist(embeddings, embeddings)
    batch_size = labels.size(0)
    negatives = []
    for i in range(batch_size):
        pos_mask = (labels == labels[i])
        neg_mask = (labels != labels[i])
        pos_dists = dist[i][pos_mask]
        anchor_positive_dist = pos_dists.mean()  # average positive distance
        candidates = [j for j in neg_mask.nonzero(as_tuple=False).view(-1).tolist()
                      if dist[i][j] > anchor_positive_dist and dist[i][j] < anchor_positive_dist + margin]
        if candidates:
            negatives.append(random.choice(candidates))
        else:
            # fallback to random negative
            negs = neg_mask.nonzero(as_tuple=False).view(-1)
            negatives.append(random.choice(negs.tolist()))
    return torch.tensor(negatives, device=labels.device)


def distance_weighted_sampling(embeddings, labels, cutoff=0.5, nonzero=1e-12):
    """Distance-weighted sampling as in Wu et al. ICCV 2017"""
    dist = torch.cdist(embeddings, embeddings)
    batch_size = labels.size(0)
    weights = []
    for i in range(batch_size):
        mask = labels != labels[i]
        d = dist[i][mask]
        # density estimation: p(d) ~ d^(dim-2) * (1 - d^2/4)^((dim-3)/2)
        dim = embeddings.size(1)
        q_d = (d + nonzero).pow(dim - 2) * (1 - 0.25 * d.pow(2)).clamp(min=0)
        w = 1.0 / (q_d + nonzero)
        w = w * (d > cutoff).float()
        w = w / w.sum()
        weights.append(w)
    negatives = []
    for i in range(batch_size):
        neg_indices = (labels != labels[i]).nonzero(as_tuple=False).view(-1)
        choice = torch.multinomial(weights[i], 1)
        negatives.append(neg_indices[choice].item())
    return torch.tensor(negatives, device=labels.device)
