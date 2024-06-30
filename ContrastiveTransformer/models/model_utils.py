import torch.nn.functional as F
import torch
import numpy as np

def create_pairs(data, max_positive_interval=180, pair_percentage=0.5):
    num_samples = data.shape[0]
    total_pairs = (num_samples * (num_samples - 1)) // 2
    num_pairs = int(total_pairs * pair_percentage)
    
    positive_pairs = []
    negative_pairs = []

    for _ in range(num_pairs):
        anchor_idx = np.random.randint(0, num_samples)
        pos_offset = np.random.randint(1, max_positive_interval + 1)
        pos_idx = (anchor_idx + pos_offset) % num_samples
        positive_pairs.append((data[anchor_idx], data[pos_idx]))

        neg_offset = np.random.randint(max_positive_interval + 1, num_samples)
        neg_idx = (anchor_idx + neg_offset) % num_samples
        negative_pairs.append((data[anchor_idx], data[neg_idx]))

    return positive_pairs, negative_pairs

def custom_contrastive_loss(features, temperature=0.5):
    batch_size = features.shape[0] // 2  # Because we have positive and negative pairs
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    loss = F.cross_entropy(logits, labels)

    return loss

