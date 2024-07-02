import os
import torch.distributed as dist
import logging


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logging.info(f'Setup process group for rank {rank} on port {port}.')

def cleanup():
    dist.destroy_process_group()
    logging.info('Destroyed process group.')

