import torch
import torch.nn.functional as F

class NT_XentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature, device, world_size=1):
        super(NT_XentLoss, self).__init__()
        # self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        # self.mask = self._get_mask(batch_size)
        self.world_size = world_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_mask(self, batch_size, world_size, device):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool, device=device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, zis, zjs):
        self.batch_size = zis.size(0)
        N = 2 * self.batch_size * self.world_size
        self.mask = self._get_mask(self.batch_size, self.world_size, self.device)

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        representations = torch.cat([zis, zjs], dim=0)

        similarity_matrix = torch.matmul(representations, representations.T)
        similarity_matrix = similarity_matrix / self.temperature

        positives = torch.cat([torch.diag(similarity_matrix, self.batch_size * self.world_size),
                               torch.diag(similarity_matrix, -self.batch_size * self.world_size)], dim=0)
        negatives = similarity_matrix[self.mask].view(N, -1)

        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)

        loss = self.criterion(logits, labels)
        loss = loss / N
        return loss
