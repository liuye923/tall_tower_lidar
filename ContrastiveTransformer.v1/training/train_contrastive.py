import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.data_loader import load_and_norm_netcdf_data
from utils.augmentations import create_pairs
from models.vision_transformer import VisionTransformer
import torch.multiprocessing as mp
import os
import logging
import argparse
import json
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-8  # Add a small constant for numerical stability

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        sim_matrix = torch.exp(sim_matrix)
        mask = torch.eye(batch_size, dtype=torch.bool).to(z_i.device)
        pos_sim = torch.diag(sim_matrix)
        loss = -torch.log(pos_sim / (sim_matrix.sum(dim=1) - pos_sim + self.epsilon))
        return loss.mean()

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

def save_checkpoint(model, optimizer, epoch, loss, config):
    save_path = os.path.join(config['save_dir'], f'model_epoch_{epoch+1}.pt')
    os.makedirs(config['save_dir'], exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    logging.info(f"Model saved at {save_path}")

def save_features(features, config):
    save_path = os.path.join(config['save_dir'], 'features.npy')
    np.save(save_path, features)
    logging.info(f"Features saved at {save_path}")

def train_contrastive(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(config['distributed_port'])
    if config['use_distributed']:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Load your data and preprocess it
    data_dir = config['data_dir']
    time_range = tuple(config['time_range'])
    data = load_and_norm_netcdf_data(data_dir, time_range)
    
    # Create pairs for contrastive learning
    pairs, labels = create_pairs(data, max_interval=180)
    pairs = np.array(pairs)
    labels = np.array(labels)
    pairs = torch.tensor(pairs)
    labels = torch.tensor(labels)
    
    # Log the shapes
    logging.info(f'Pairs shape: {pairs.shape}')
    logging.info(f'Labels shape: {labels.shape}')
    
    # Create dataloader
    dataset = TensorDataset(pairs, labels)
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize models and optimizer
    model = VisionTransformer(
        image_size=(37, 45),   # dimensions of your data
        patch_size=(37, 45),   # using the full image as a patch
        num_classes=64,
        dim=512,               # hidden_dim
        depth=6,               # num_layers
        heads=8,               # num_heads
        mlp_dim=1024,
        channels=6,            # number of variables
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    )
    device = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    model.to(device)
    if config['use_data_parallel']:
        model = nn.DataParallel(model)
    if config['use_distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    all_features = []

    for epoch in range(config['epochs']):
        epoch_features = []
        for batch in train_loader:
            inputs, targets = batch
            # inputs shape: (batch_size, 2, 6, 37, 45)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            z_i, z_j = model(inputs[:, 0]), model(inputs[:, 1])  # Process each sample independently

            # Log the shapes of z_i and z_j
            logging.info(f'z_i shape: {z_i.shape}, z_j shape: {z_j.shape}')

            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            if rank == 0:
                logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

            # Collect features for K-means
            epoch_features.append(z_i.detach().cpu().numpy())
            epoch_features.append(z_j.detach().cpu().numpy())

        # Concatenate features for this epoch
        epoch_features = np.concatenate(epoch_features, axis=0)
        all_features.append(epoch_features)

        # Save the model after each epoch
        if rank == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), config)

    if rank == 0:
        all_features = np.concatenate(all_features, axis=0)
        # Remove duplicates
        unique_features = np.unique(all_features, axis=0)
        logging.info(f"Unique features shape: {unique_features.shape}")
        save_features(unique_features, config)

    if config['use_distributed']:
        torch.distributed.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train Contrastive Learning Model")
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    log_level = getattr(logging, config['log_level'].upper(), logging.INFO)
    setup_logging(log_level)

    if config['use_gpu'] and torch.cuda.is_available():
        device = 'cuda'
        world_size = torch.cuda.device_count()
    else:
        device = 'cpu'
        world_size = torch.multiprocessing.cpu_count() if config['use_distributed'] else 1

    if config['use_distributed']:
        mp.spawn(train_contrastive,
                 args=(world_size, config),
                 nprocs=world_size,
                 join=True)
    else:
        train_contrastive(0, world_size, config)

if __name__ == "__main__":
    main()
