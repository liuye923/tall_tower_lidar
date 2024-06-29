import torch
import torch.nn as nn
from .mae import MAE
from .vit import ViT
from .vit import Transformer, Attention, FeedForward
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import TransformerDecoder, TransformerDecoderLayer


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ERAencoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ERAencoder,self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :]

        x = self.dropout(x)

        x = self.transformer(x)

        return x


class ERAdecoder(nn.Module):
    def __init__(self, *, final_image_size, patch_size, dim, depth, heads, mlp_dim, final_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ERAdecoder,self).__init__()

        image_height, image_width = pair(final_image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = final_channels * patch_height * patch_width

        transformer_layer = TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first = True)
        self.transformer = TransformerDecoder(transformer_layer, num_layers=depth)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1056, dim))

        self.mlp_head = nn.Linear(dim, patch_dim)

    def forward(self, x, memory):

        x += self.pos_embedding[:,:] #added for version 3

        x = self.transformer(x, memory)
        
        x = x[:,:910]   #take only the first 910 tokens

        return self.mlp_head(x)



class ERA5Upscaler(nn.Module):

    def __init__(self, encoder, hrrr_decoder):
        super(ERA5Upscaler,self).__init__()

        self.encoder = encoder
        self.hrrr_decoder = hrrr_decoder


    def forward(self, img, img_target):

        """ Encoder """
        x = self.encoder.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.encoder.pos_embedding[:, :]
        x = self.encoder.dropout(x)
        memory = self.encoder.transformer(x)

        """ Decoder """
        x = torch.zeros_like(x)
        x += self.hrrr_decoder.pos_embedding[:,:]
        x = self.hrrr_decoder.transformer(x, memory)
        
        x = x[:,:910]   #take only the first 910 tokens

        return self.hrrr_decoder.mlp_head(x), img_target


if __name__ == '__main__':
    

    enc = ERAencoder(image_size=(240,440),
	       patch_size=(15,22),
	       num_classes=192,
           channels=12,
	       dim=768,
	       depth=8,
	       heads=4,
	       mlp_dim=3)


    mae = MAE(encoder=enc, decoder_dim=192)
    

    sample = torch.rand(10,12,240,440)

    y = mae(sample)

    print(y)

    #model = ERA5Upscaler(encoder=enc, decoder=enc)




