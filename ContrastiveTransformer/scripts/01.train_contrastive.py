import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import logging
import argparse
from data.data_loader import CustomDataset, load_data
from data.sample_data import prepare_met
from models.transformer_encoder import TransformerEncoder
from models.projection_head import ProjectionHead
from models.model_utils import NT_XentLoss
from utils.config import read_config
from utils.logging import set_logging_level
from utils.distributed import setup, cleanup
from utils.checkpoint import save_checkpoint, load_checkpoint
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter

import time

def train_contrastive_model(
        rank, world_size, data, 
        num_epochs=20, batch_size=32, learning_rate=0.001, use_distributed=False, 
        checkpoint_path='checkpoint.pth', resume_checkpoint=True, 
        port=9330, seed=330,
        model_encoder_path=None, model_projection_path=None,
    ):
    # set random status
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = False


    if use_distributed:
        setup(rank, world_size, port)
        if torch.cuda.is_available(): 
            torch.cuda.set_device(rank)

    # Example parameters, adjust as necessary
    image_size = (92, 112)  # Example dimensions of your data
    patch_size = (92, 112)  # Set patch_size to full image for testing
    num_layers = 6
    num_heads = 8
    hidden_dim = 512
    channels = 6
    projection_dim = 128

    model = TransformerEncoder(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, channels=channels)
    projection_head = ProjectionHead(input_dim=hidden_dim, projection_dim=projection_dim)
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        model.to(device)
        projection_head.to(device)
        if use_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            projection_head = torch.nn.parallel.DistributedDataParallel(projection_head, device_ids=[rank])
    else:
        device = torch.device('cpu')
        model.to(device)
        projection_head.to(device)
        if use_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model)
            projection_head = torch.nn.parallel.DistributedDataParallel(projection_head)

    if use_distributed:
        criterion = NT_XentLoss(batch_size=batch_size, temperature=0.5, device=device, world_size=world_size)
    else:
        criterion = NT_XentLoss(batch_size=batch_size, temperature=0.5, device=device, world_size=1)
       
    optimizer = optim.Adam(list(model.parameters()) + list(projection_head.parameters()), lr=learning_rate)

    start_epoch = 0
    if resume_checkpoint:
        start_epoch = load_checkpoint(checkpoint_path, model, projection_head, optimizer)

    logging.info(f'Starting training from epoch {start_epoch} with {num_epochs} epochs, batch size {batch_size}, learning rate {learning_rate}.')

    # Create custom dataset and sampler
    dataset = CustomDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if use_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if use_distributed else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=(val_sampler is None), num_workers=4, pin_memory=True)

    logging.info(f"Rank {rank}: Dataset size: {len(train_dataset)}, Train_DataLoader length: {len(train_loader)}")
    logging.info(f"Rank {rank}: Dataset size: {len(val_dataset)}, Val_DataLoader length: {len(val_loader)}")

    writer = SummaryWriter('scripts/contrastive_learning_experiment')

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        if use_distributed:
            train_sampler.set_epoch(epoch)
        
        logging.info(f"Rank {rank}: Starting epoch {epoch+1}/{num_epochs}")
        
        # Adjust to handle batches correctly
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=rank if use_distributed else 0) as pbar:
            batch_times = []
            model.train()
            projection_head.train()
            train_loss = 0.
            for batch_idx, batch_data in enumerate(pbar):
                batch_start_time = time.time()

                logging.debug(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx+1}: {batch_data[0].shape}")
                
                x_i = batch_data[0].to(device)
                x_j = batch_data[1].to(device)

                # Forward pass through the model and projection head
                h_i = model(x_i)   # size of [batch_size, hidden_dim]
                h_j = model(x_j)
                z_i = projection_head(h_i) # size of [batch_size, projection_dim]
                z_j = projection_head(h_j)

                # Compute contrastive loss
                loss = criterion(z_i, z_j)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

                pbar.set_postfix(loss=loss.item(), batch_time=batch_time)
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
        # model.eval()
        # projection_head.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for batch_idx, batch_data in enumerate(val_loader):  
        #         x_i = batch_data[0].to(device)
        #         x_j = batch_data[1].to(device)

        #         # Forward pass through the model and projection head
        #         h_i = model(x_i)   # size of [batch_size, hidden_dim]
        #         h_j = model(x_j)
        #         z_i = projection_head(h_i) # size of [batch_size, projection_dim]
        #         z_j = projection_head(h_j)

        #         # Compute contrastive loss
        #         loss = criterion(z_i, z_j)
        #         val_loss += loss.item()
        #     val_loss /= len(val_loader)

        with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Eval", position=rank if use_distributed else 0) as pbar:
            model.eval()
            projection_head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(pbar):  
                    x_i = batch_data[0].to(device)
                    x_j = batch_data[1].to(device)

                    # Forward pass through the model and projection head
                    h_i = model(x_i)   # size of [batch_size, hidden_dim]
                    h_j = model(x_j)
                    z_i = projection_head(h_i) # size of [batch_size, projection_dim]
                    z_j = projection_head(h_j)

                    # Compute contrastive loss
                    loss = criterion(z_i, z_j)
                    pbar.set_postfix(loss=loss.item(), batch_time=batch_time)
                    val_loss += loss.item()
                val_loss /= len(val_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)


        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        logging.info(f"Rank {rank}: Epoch {epoch+1} completed in {epoch_time:.2f} seconds, average batch time: {avg_batch_time:.2f} seconds")

        if rank == 0 or not use_distributed:
            # logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")
            logging.info(f"Epoch {epoch+1}, Train_Loss: {train_loss}, Val_Loss: {val_loss}")
            save_checkpoint(epoch + 1, model, projection_head, optimizer, checkpoint_path)

    if rank == 0 or not use_distributed:
        writer.close()
        torch.save(model.state_dict(), model_encoder_path)
        torch.save(projection_head.state_dict(), model_projection_path)
        logging.info('Saved model and projection head state dicts.')

    if use_distributed:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contrastive learning model.")
    parser.add_argument('--config', type=str, default='config.single.test.json', help='Path to the configuration file')

    args = parser.parse_args()
    
    # Loading configures
    config = read_config(args.config)

    set_logging_level(config.get('logging_level', 'INFO'))

    use_distributed = config['use_distributed']
    world_size = int(config['world_size'])
    num_epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    checkpoint_path = config['checkpoint_path']
    resume_checkpoint = config['resume_checkpoint']
    port = config.get('port', 9330)
    seed = config.get('seed', 330)
    model_encoder_path = config['model_encoder_path']
    model_projection_path = config['model_projection_path']

    prepare_train_data = config['prepare_train_data']

    logging.info(f"Read configuration from {args.config}")
    logging.info(json.dumps(config, indent=2))

    # prepare data
    if prepare_train_data:
        prepare_met(
            data_dir=config['raw_data_dir'], 
            save_path=config['train_data_path'],
            save_idx_path=config['train_data_index_path'],
            time_range=config['train_period'],
            file_suffix=config['raw_file_suffix'],
            range_path=config['raw_range_path'],
        )
        logging.info(f'Prepare data {config["train_data_path"]} with shape.')

    data = load_data(config['train_data_path'])
    logging.info(f'Loaded data from {config["train_data_path"]} with shape {data.shape}.')

    if use_distributed:
        import torch.multiprocessing as mp
        mp.spawn(train_contrastive_model,
                 args=(world_size, data, num_epochs, batch_size, learning_rate, True, checkpoint_path, resume_checkpoint, port),
                 nprocs=world_size,
                 join=True)
    else:
        train_contrastive_model(0, 1, data, num_epochs, batch_size, learning_rate, False, checkpoint_path, resume_checkpoint, port, seed, model_encoder_path, model_projection_path)
