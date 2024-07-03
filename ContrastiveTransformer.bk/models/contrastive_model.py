import torch.nn as nn

class ContrastiveModel(nn.Module):
    def __init__(self, encoder, projector=None):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x):
        h = self.encoder(x)
        if self.projector is None:
            return h
        else:
            z = self.projector(h)
            return h, z
