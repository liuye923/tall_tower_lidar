import torch
import torch.nn as nn
from vit_pytorch import ViT

class TransformerEncoder(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, channels):
        super(TransformerEncoder, self).__init__()
        self.encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=hidden_dim,  # Output dimension of the ViT
            dim=hidden_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=hidden_dim * 4,
            pool='cls',
            channels=channels
        )
    
    def forward(self, x):
        return self.encoder(x.type(torch.FloatTensor)).type(torch.FloatTensor)
