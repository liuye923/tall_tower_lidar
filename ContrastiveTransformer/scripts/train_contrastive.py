import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import logging
import argparse
from data.data_loader import CustomDataset, load_data
from models.transformer_encoder import TransformerEncoder
from models.projection_head import ProjectionHead
from models.model_utils import custom_contrastive_loss
from utils.config import read_config
from utils.logging import set_logging_level
from utils.distributed import setup, cleanup
from utils.checkpoint import save_checkpoint, load_checkpoint
import time
import numpy as np

def create_pairs(data, max_positive_interval=180, pair_percentage=0.5):
    num_samples = data.size(0)
    num_pairs = int(num_samples * pair_percentage)
    
    positive_pairs = []
    negative_pairs = []
    labels = []

    for _ in range(num_pairs):
        idx1 = np.random.randint(0, num_samples)
        idx2 = idx1 + np.random.randint(1, max_positive_interval + 1)
        
        if idx2 < num_samples:
            positive_pairs.append((data[idx1], data[idx2]))
            labels.append(1)
        
        idx2 = np.random.randint(0, num_samples)
        while abs(idx1 - idx2) <= max_positive_interval:
            idx2 = np.random.randint(0, num_samples)
        
        negative_pairs.append((data[idx1], data[idx2]))
        labels.append(0)
    
    pairs = positive_pairs + negative_pairs
    labels = torch.tensor(labels, dtype=torch.float32)
    pairs = torch.stack([torch.stack(pair) for pair in pairs])

    return pairs, labels

def train_contrastive_model(rank, world_size, data, num_epochs=100, batch_size=32, learning_rate=0.001, use_distributed=False, checkpoint_path='checkpoint.pth', resume_checkpoint=True, port=9330):
    if use_distributed:
        setup(rank, world_size, port)
    
    # Example parameters, adjust as necessary
    image_size = (37, 45)  # Example dimensions of your data
    patch_size = (37, 45)  # Set patch_size to full image for testing
    num_layers = 6
    num_heads = 8
    hidden_dim = 512
    channels = 6

    torch.manual_seed(0)
    model = TransformerEncoder(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, channels=channels)
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        model.to(device)
        projection_head = ProjectionHead(input_dim=hidden_dim, projection_dim=128).to(device)
        if use_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            projection_head = torch.nn.parallel.DistributedDataParallel(projection_head, device_ids=[rank])
    else:
        device = torch.device('cpu')
        projection_head = ProjectionHead(input_dim=hidden_dim, projection_dim=128)
        if use_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model)
            projection_head = torch.nn.parallel.DistributedDataParallel(projection_head)
    
    optimizer = optim.Adam(list(model.parameters()) + list(projection_head.parameters()), lr=learning_rate)

    start_epoch = 0
    if resume_checkpoint:
        start_epoch = load_checkpoint(checkpoint_path, model, projection_head, optimizer)

    logging.info(f'Starting training from epoch {start_epoch} with {num_epochs} epochs, batch size {batch_size}, learning rate {learning_rate}.')

    # Create custom dataset and sampler
    dataset = CustomDataset(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if use_distributed else None
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)
    
    logging.info(f"Rank {rank}: Dataset size: {len(dataset)}, DataLoader length: {len(dataloader)}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        if use_distributed:
            sampler.set_epoch(epoch)
        
        logging.info(f"Rank {rank}: Starting epoch {epoch+1}/{num_epochs}")
        
        # Adjust to handle batches correctly
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", position=rank if use_distributed else 0) as pbar:
            batch_times = []
            for batch_idx, batch_data in enumerate(pbar):
                batch_start_time = time.time()
                logging.debug(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx+1}: {batch_data.shape}")
                
                batch_data = batch_data.to(device)

                # Ensure the input data is reshaped correctly for the ViT model
                batch_data = batch_data.view(-1, channels, *image_size)  # Reshape to (batch_size*2, channels, height, width)

                # Generate pairs for contrastive learning
                pairs, labels = create_pairs(batch_data)

                # Forward pass through the model and projection head
                encoded_features = model(pairs)
                projected_features = projection_head(encoded_features)

                # Compute contrastive loss
                loss = custom_contrastive_loss(projected_features, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

                pbar.set_postfix(loss=loss.item(), batch_time=batch_time)

        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        logging.info(f"Rank {rank}: Epoch {epoch+1} completed in {epoch_time:.2f} seconds, average batch time: {avg_batch_time:.2f} seconds")

        if rank == 0 or not use_distributed:
            logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")
            save_checkpoint(epoch + 1, model, projection_head, optimizer, checkpoint_path)

    if rank == 0 or not use_distributed:
        torch.save(model.state_dict(), 'models/transformer_encoder.pth')
        torch.save(projection_head.state_dict(), 'models/projection_head.pth')
        logging.info('Saved model and projection head state dicts.')

    if use_distributed:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contrastive learning model.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--port', type=int, default=9330, help='Port for distributed training')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')

    args = parser.parse_args()

    config = read_config(args.config)
    set_logging_level(args.log_level)

    data = load_data(config['data_path'])
    logging.info(f'Loaded data from {config["data_path"]} with shape {data.shape}.')

    use_distributed = config['use_distributed']
    world_size = int(config['world_size'])
    num_epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    checkpoint_path = config['checkpoint_path']
    resume_checkpoint = config['resume_checkpoint']
    port = args.port

    if use_distributed:
        import torch.multiprocessing as mp
        mp.spawn(train_contrastive_model,
                 args=(world_size, data, num_epochs, batch_size, learning_rate, True, checkpoint_path, resume_checkpoint, port),
                 nprocs=world_size,
                 join=True)
    else:
        train_contrastive_model(0, 1, data, num_epochs, batch_size, learning_rate, False, checkpoint_path, resume_checkpoint, port)
