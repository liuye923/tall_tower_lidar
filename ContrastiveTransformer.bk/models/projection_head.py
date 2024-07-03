import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        return self.projection(x)
