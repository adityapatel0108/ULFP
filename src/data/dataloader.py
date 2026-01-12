"""
DataLoader configurations for face recognition datasets
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler
from .dataset import MS1MV2Dataset, LFWDataset


def get_ms1mv2_dataloader(root_dir, batch_size=128, num_workers=4, 
                          is_training=True, shuffle=None, distributed=False):
    """
    Get DataLoader for MS1MV2 dataset.
    
    Args:
        root_dir: Root directory of MS1MV2 dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        is_training: Whether this is training data
        shuffle: Whether to shuffle (defaults to is_training)
        distributed: Whether using distributed training
    
    Returns:
        DataLoader instance
    """
    dataset = MS1MV2Dataset(root_dir, is_training=is_training)
    
    if shuffle is None:
        shuffle = is_training
    
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training
    )
    
    return dataloader


def get_lfw_dataloader(root_dir, pairs_file, batch_size=64, num_workers=4):
    """
    Get DataLoader for LFW dataset.
    
    Args:
        root_dir: Root directory of LFW dataset
        pairs_file: Path to pairs file
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    dataset = LFWDataset(root_dir, pairs_file)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_verification_dataloader(dataset_class, root_dir, pairs_file=None, 
                                batch_size=64, num_workers=4):
    """
    Get DataLoader for verification datasets.
    
    Args:
        dataset_class: Dataset class to use
        root_dir: Root directory of dataset
        pairs_file: Path to pairs/protocol file
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    if pairs_file:
        dataset = dataset_class(root_dir, pairs_file)
    else:
        dataset = dataset_class(root_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
