"""
Title: CIFAR100_loader.py
"""

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from typing import Callable, Optional
from logging import getLogger
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

_GLOBAL_SEED = 0
logger = getLogger()



def make_cifar100(transform: Optional[Callable]=None,
                  batch_size: int=128,
                  pin_mem: bool=True,
                  num_workers: int=8,
                  world_size: int=1,
                  rank: int=0,
                  root_path: str='/localscratch/hsun409/',
                  image_folder: str='data/',
                  training: bool=False,
                  drop_last: bool=True):

    # Initialize the dataset
    dataset = CIFAR100(root=Path(root_path) / image_folder,
                        train=training,
                        transform=transform,
                        download=True)
    
    logger.info('CIFAR100 dataset created')

    # Set the distributed sampler
    dist_sampler = DistributedSampler(dataset=dataset,
                                      num_replicas=world_size,
                                      rank=rank)

    data_loader = DataLoader(dataset=dataset, 
                             batch_size=batch_size,
                             num_workers=num_workers,
                             drop_last=drop_last, 
                             pin_memory=pin_mem, 
                             sampler=dist_sampler,
                             persistent_workers=False)
    
    logger.info('CIFAR100 data loader created.')
    return dataset, data_loader, dist_sampler
