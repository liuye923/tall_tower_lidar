import torch
import torch.nn as nn
from vit_pytorch import ViT
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Dummy data generation function
def generate_dummy_data(time_steps, channels, height, width):
    return np.random.rand(time_steps, channels, height, width)

class ViTFeatureExtractor(nn.Module):
    def __init__(self, image_size=(40, 45), patch_size=5, dim=512, depth=6, heads=8, mlp_dim=2048, channels=6):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=dim,  # Output dimension of the ViT
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels
        )
    
    def forward(self, x):
        return self.vit(x)

class ViT_BiLSTM_FeatureExtractor(nn.Module):
    def __init__(self, image_size=(40, 45), patch_size=5, hidden_dim=512, num_layers=2):
        super(ViT_BiLSTM_FeatureExtractor, self).__init__()
        self.feature_extractor = ViTFeatureExtractor(image_size=image_size, patch_size=patch_size)
        
        # Bidirectional LSTM to model temporal dependencies
        self.bilstm = nn.LSTM(
            input_size=512,  # This should match the output dimension of ViT
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, x):
        batch_size, num_channels, seq_length, height, width = x.size()
        
        # Reshape to (batch_size * seq_length, num_channels, height, width)
        x = x.view(batch_size * seq_length, num_channels, height, width)
        
        # Feature extraction using ViT
        features = self.feature_extractor(x)
        
        # Reshape to (batch_size, seq_length, feature_dim)
        features = features.view(batch_size, seq_length, -1)
        
        # BiLSTM for temporal dependencies
        bilstm_out, _ = self.bilstm(features)
        
        return bilstm_out

# Ensure the input image size is compatible with the patch size
def pad_image_to_fit_patch_size(data, patch_size):
    _, _, height, width = data.shape
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size
    padded_data = np.pad(data, ((0, 0), (0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    return padded_data

# Generate dummy data
time_steps = 109
channels = 6
height = 37
width = 45

dummy_data = generate_dummy_data(time_steps, channels, height, width)
dummy_data = pad_image_to_fit_patch_size(dummy_data, patch_size=5)

# Convert dummy_data to torch tensor and ensure the correct shape
input_data = torch.tensor(dummy_data, dtype=torch.float32)

# Ensure input_data has shape (batch_size, num_channels, seq_length, height, width)
input_data = input_data.unsqueeze(0)  # Add batch dimension

# Example usage to extract and store hidden states
model = ViT_BiLSTM_FeatureExtractor(image_size=(40, 45), patch_size=5)

# Process data in batches using tqdm for progress tracking
batch_size = 1
num_batches = input_data.size(0) // batch_size

logging.info(f"Starting feature extraction for {num_batches} batches")
all_hidden_states = []

for i in tqdm(range(num_batches), desc="Processing Batches"):
    batch_data = input_data[i*batch_size:(i+1)*batch_size]
    print(batch_data.shape)
    hidden_states = model(batch_data)
    all_hidden_states.append(hidden_states)

# Concatenate all hidden states
all_hidden_states = torch.cat(all_hidden_states, dim=0)

# Save hidden states for later use
torch.save(all_hidden_states, 'hidden_states.pt')
logging.info("Hidden states saved to hidden_states.pt")

