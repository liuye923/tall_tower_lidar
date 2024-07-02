import numpy as np
import torch

def create_probability_distribution(n, inverse=False):
    prob = np.arange(n, 0, -1) if not inverse else np.arange(1, n+1)
    prob = prob / prob.sum()
    return prob

def sample_pair(index, data, max_interval=180, is_positive=True):
    n = len(data)
    if is_positive:
        start = max(0, index - max_interval)
        end = min(n, index + max_interval + 1)
        candidates = list(range(start, index)) + list(range(index + 1, end))
        prob = create_probability_distribution(len(candidates))
    else:
        candidates = list(range(0, max(0, index - max_interval))) + list(range(min(n, index + max_interval + 1), n))
        prob = create_probability_distribution(len(candidates), inverse=True)
    
    sampled_index = np.random.choice(candidates, p=prob)
    return data[sampled_index]

def create_pairs(data, max_interval=180):
    pairs = []
    labels = []
    num_samples = len(data)
    
    for i in range(num_samples):
        positive_sample = sample_pair(i, data, max_interval, is_positive=True)
        pairs.append((data[i], positive_sample))
        labels.append(1)
        
        negative_sample = sample_pair(i, data, max_interval, is_positive=False)
        pairs.append((data[i], negative_sample))
        labels.append(0)
    
    return pairs, labels
