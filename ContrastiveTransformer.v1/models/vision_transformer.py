import torch
import torch.nn as nn
from vit_pytorch import ViT

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=6, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.vit = ViT(
            image_size = image_size,  # (37, 45)
            patch_size = patch_size,  # (37, 45), since we're using the full image as a patch
            num_classes = num_classes,
            dim = dim,  # hidden_dim
            depth = depth,  # num_layers
            heads = heads,  # num_heads
            mlp_dim = mlp_dim,
            pool = 'cls',
            channels = channels,
            dim_head = dim_head,
            dropout = dropout,
            emb_dropout = emb_dropout
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        return self.vit(x)
