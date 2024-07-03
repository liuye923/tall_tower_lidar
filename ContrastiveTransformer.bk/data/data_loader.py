import torch
import random
from torch.utils.data import Dataset
import numpy as np
import logging
from skimage.transform import rescale as skrescale
from scipy import signal as ssignal

class CustomDataset(Dataset):
    def __init__(self, data, batch_size=32, time_window=180, seed=None):
        self.data = data
        self.batch_size = batch_size
        self.cut_data_to_batch_size()
        self.time_window = time_window
        self.dataset = data
        self.seed = seed
        self.mean5kernel = np.ones((5,5))/25

        
        if self.seed is not None:
            self._set_seed(self.seed)

    def cut_data_to_batch_size(self):
        length = (len(self.data) // self.batch_size) * self.batch_size
        self.data = self.data[:length]
    
    def _set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def __len__(self):
        return len(self.data)
    
    def _select_positive_idx(self, idx):
        # Bidirectional window
        start_idx = max(0, idx - self.time_window)
        end_idx = min(len(self.data) - 1, idx + self.time_window)
        
        possible_indices = list(range(start_idx, idx)) + list(range(idx + 1, end_idx + 1))
        
        if not possible_indices:
            return None
        
        # Create a probability distribution that favors closer time stamps
        distances = np.abs(np.array(possible_indices) - idx)
        probabilities = (1 / (distances + 1))
        probabilities /= probabilities.sum()

        positive_idx = np.random.choice(possible_indices, p=probabilities)
        return positive_idx

    def _preprocess(self, data):
        preprocessed_data = []

        for channel in data:
            # Step 1: Rescale
            data_step1 = skrescale(channel, (2.5, 2.5), anti_aliasing=True)
            # Step 2: Apply 5x5 mean filter
            _preprocessed_data = ssignal.convolve2d(data_step1, self.mean5kernel, boundary='symm', mode='same')

            preprocessed_data.append(_preprocessed_data)
        logging.debug(f'CustomDataset - datashape: {data_step1.shape}')
        logging.debug(f'CustomDataset - datashape: {_preprocessed_data.shape}')

        return np.stack(preprocessed_data, axis=0)

    def __getitem__(self, idx):
        positive_idx = self._select_positive_idx(idx)
        logging.debug(f'CustomDataset - paired index: {idx}, {positive_idx}, {positive_idx-idx}')
        return self._preprocess(self.data[idx]), self._preprocess(self.data[positive_idx])

def load_data(data_path):
    data = np.load(data_path)
    return data

