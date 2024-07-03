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
import time

def train_contrastive_model(
    rank, world_size, data, num_epochs=20, batch_size=32, learning_rate=0.001, 
    use_distributed=False, checkpoint_path=None, resume_checkpoint=True, 
    port=9330, 
    seed=330, 
    model_encoder_path=None, model_projection_path=None,
    embedding_encoder_path=None, embedding_projection_path=None,
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

    model.load_state_dict(torch.load(model_encoder_path))
    logging.info(f'Load pretrained encoder from {model_encoder_path}.')

    projection_head.load_state_dict(torch.load(model_projection_path))
    logging.info(f'Load pretrained projection_head from {model_projection_path}.')

    # Create custom dataset and sampler
    dataset = CustomDataset(data)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if use_distributed else None
    # data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Rank {rank}: Dataset size: {len(dataset)}, Train_DataLoader length: {len(data_loader)}")

    epoch_start_time = time.time()
    all_encoder = []
    all_projection = []
    # Adjust to handle batches correctly
    with tqdm(data_loader, desc=f"Evaluation", position=rank if use_distributed else 0) as pbar:
        batch_times = []
        model.eval()
        projection_head.eval()
        eval_loss = 0.
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                batch_start_time = time.time()

                logging.debug(f"Rank {rank}, Batch {batch_idx+1}: {batch_data[0].shape}")
                
                x_i = batch_data[0].to(device)
                # x_j = batch_data[1].to(device)

                # Forward pass through the model and projection head
                h_i = model(x_i)   # size of [batch_size, hidden_dim]
                # h_j = model(x_j)
                z_i = projection_head(h_i) # size of [batch_size, projection_dim]
                # z_j = projection_head(h_j)

                # Compute contrastive loss
                # loss = criterion(z_i, z_j)
                all_encoder.append(h_i)
                all_projection.append(z_i)

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

                # pbar.set_postfix(loss=loss.item(), batch_time=batch_time)
                pbar.set_postfix(batch_time=batch_time)
            #     eval_loss += loss.item()
            # eval_loss /= len(data_loader)
    all_encoder = torch.cat(all_encoder, dim=0)
    all_projection = torch.cat(all_projection, dim=0)

    epoch_time = time.time() - epoch_start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    logging.info(f"Rank {rank}: Evaluation completed in {epoch_time:.2f} seconds, average batch time: {avg_batch_time:.2f} seconds")

    if use_distributed:
        torch.distributed.barrier()  # Make sure all computing completed

    if rank == 0:
        torch.save(all_encoder, embedding_encoder_path)
        torch.save(all_projection, embedding_projection_path)
        logging.info('Saved all_encoder and all_projection to files.')    

    if use_distributed:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contrastive learning model.")
    parser.add_argument('--config', type=str, default='config.single.test.json', help='Path to the configuration file')

    args = parser.parse_args()
    
    # Loading configures
    config = read_config(args.config)

    print(config.get('logging_level', 'INFO'))
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
    embedding_encoder_path = config['embedding_encoder_path']
    embedding_projection_path = config['embedding_projection_path']

    prepare_production_data = config['prepare_production_data']

    logging.info(f"Read configuration from {args.config}")
    logging.info(json.dumps(config, indent=2))

    # prepare data
    if prepare_production_data:
        prepare_met(
            data_dir=config['raw_data_dir'], 
            save_path=config['production_data_path'],
            save_idx_path=config['production_data_index_path'],
            time_range=config['production_period']
        )
        logging.info(f'Prepare data {config["production_data_path"]} with shape.')

    data = load_data(config['production_data_path'])
    logging.info(f'Loaded data from {config["production_data_path"]} with shape {data.shape}.')

    if use_distributed:
        import torch.multiprocessing as mp
        mp.spawn(train_contrastive_model,
                 args=(world_size, data, num_epochs, batch_size, learning_rate, True, checkpoint_path, resume_checkpoint, port),
                 nprocs=world_size,
                 join=True)
    else:
        train_contrastive_model(0, 1, data, num_epochs, batch_size, learning_rate, False, checkpoint_path, resume_checkpoint, port, seed, model_encoder_path, model_projection_path, embedding_encoder_path, embedding_projection_path)
