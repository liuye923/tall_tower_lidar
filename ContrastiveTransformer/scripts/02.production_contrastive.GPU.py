import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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
from utils.checkpoint import save_checkpoint, load_checkpoint
import numpy as np
import json
import time

def train_contrastive_model(
    data, num_epochs=20, batch_size=32, learning_rate=0.001, 
    checkpoint_path=None, resume_checkpoint=True, 
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Example parameters, adjust as necessary
    image_size = (92, 112)  # Example dimensions of your data
    # patch_size = (92, 112)  # Set patch_size to full image for testing
    patch_size = (23, 28)  # Set patch_size to full image for testing
    num_layers = 6
    num_heads = 8
    hidden_dim = 512
    channels = 6
    projection_dim = 128

    model = TransformerEncoder(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, channels=channels)
    projection_head = ProjectionHead(input_dim=hidden_dim, projection_dim=projection_dim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    projection_head.to(device)

    criterion = NT_XentLoss(batch_size=batch_size, temperature=0.5, device=device, world_size=1)
    optimizer = optim.Adam(list(model.parameters()) + list(projection_head.parameters()), lr=learning_rate)

    model.load_state_dict(torch.load(model_encoder_path))
    logging.info(f'Loaded pretrained encoder from {model_encoder_path}.')

    projection_head.load_state_dict(torch.load(model_projection_path))
    logging.info(f'Loaded pretrained projection_head from {model_projection_path}.')

    # Create custom dataset and data loader
    dataset = CustomDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Dataset size: {len(dataset)}, DataLoader length: {len(data_loader)}")

    epoch_start_time = time.time()
    all_encoder = []
    all_projection = []

    # Adjust to handle batches correctly
    with tqdm(data_loader, desc=f"Evaluation") as pbar:
        batch_times = []
        model.eval()
        projection_head.eval()
        eval_loss = 0.
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                batch_start_time = time.time()

                logging.debug(f"Batch {batch_idx+1}: {batch_data[0].shape}")
                
                x_i = batch_data[0].to(device)

                # Forward pass through the model and projection head
                h_i = model(x_i)   # size of [batch_size, hidden_dim]
                z_i = projection_head(h_i) # size of [batch_size, projection_dim]

                all_encoder.append(h_i)
                all_projection.append(z_i)

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

                pbar.set_postfix(batch_time=batch_time)

    all_encoder = torch.cat(all_encoder, dim=0)
    all_projection = torch.cat(all_projection, dim=0)

    epoch_time = time.time() - epoch_start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    logging.info(f"Evaluation completed in {epoch_time:.2f} seconds, average batch time: {avg_batch_time:.2f} seconds")

    torch.save(all_encoder, embedding_encoder_path)
    torch.save(all_projection, embedding_projection_path)
    logging.info('Saved all_encoder and all_projection to files.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contrastive learning model.")
    parser.add_argument('--config', type=str, default='config.single.test.json', help='Path to the configuration file')

    args = parser.parse_args()
    
    # Loading configurations
    config = read_config(args.config)

    set_logging_level(config.get('logging_level', 'INFO'))

    num_epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    checkpoint_path = config['checkpoint_path']
    resume_checkpoint = config['resume_checkpoint']
    seed = config.get('seed', 330)
    model_encoder_path = config['model_encoder_path']
    model_projection_path = config['model_projection_path']
    embedding_encoder_path = config['embedding_encoder_path']
    embedding_projection_path = config['embedding_projection_path']

    prepare_production_data = config['prepare_production_data']

    logging.info(f"Read configuration from {args.config}")
    logging.info(json.dumps(config, indent=2))

    # Prepare data
    if prepare_production_data:
        prepare_met(
            data_dir=config['raw_data_dir'], 
            save_path=config['production_data_path'],
            save_idx_path=config['production_data_index_path'],
            time_range=config['production_period'],
            file_suffix=config['raw_file_suffix'],
            range_path=config['raw_range_path'],
       )
        logging.info(f'Prepared data {config["production_data_path"]}.')

    data = load_data(config['production_data_path'])
    logging.info(f'Loaded data from {config["production_data_path"]} with shape {data.shape}.')

    train_contrastive_model(data, num_epochs, batch_size, learning_rate, checkpoint_path, resume_checkpoint, seed, model_encoder_path, model_projection_path, embedding_encoder_path, embedding_projection_path)
