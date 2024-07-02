import torch
import os
import logging

def save_checkpoint(epoch, model, projection_head, optimizer, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'projection_head_state_dict': projection_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    logging.info(f'Saved checkpoint: {checkpoint_path}')

def load_checkpoint(checkpoint_path, model, projection_head, optimizer):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # Adjust the state dictionary keys if using DistributedDataParallel
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logging.info(f'Loaded checkpoint: {checkpoint_path} (epoch {epoch})')
        return epoch
    else:
        logging.warning(f'No checkpoint found at: {checkpoint_path}')
        return 0

