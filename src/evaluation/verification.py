"""
Face verification evaluation on standard benchmarks
"""

import torch
import numpy as np
from ..data.dataloader import get_lfw_dataloader, get_verification_dataloader
from .metrics import compute_all_metrics


def evaluate_verification(model, dataloader, device='cuda'):
    """
    Evaluate face verification on a dataset.
    
    Args:
        model: Face recognition model
        dataloader: DataLoader with (img1, img2, is_same) tuples
        device: Device to run on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_embeddings1 = []
    all_embeddings2 = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, is_same in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Extract embeddings
            embeddings1 = model.extract_features(img1)
            embeddings2 = model.extract_features(img2)
            
            all_embeddings1.append(embeddings1.cpu())
            all_embeddings2.append(embeddings2.cpu())
            all_labels.append(is_same.cpu())
    
    # Concatenate all results
    embeddings1 = torch.cat(all_embeddings1, dim=0)
    embeddings2 = torch.cat(all_embeddings2, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_all_metrics(embeddings1, embeddings2, labels)
    
    return metrics


def evaluate_lfw(model, root_dir, pairs_file, device='cuda', batch_size=64):
    """
    Evaluate on LFW dataset.
    
    Args:
        model: Face recognition model
        root_dir: Root directory of LFW dataset
        pairs_file: Path to LFW pairs file
        device: Device to run on
        batch_size: Batch size
    
    Returns:
        Dictionary with evaluation metrics
    """
    dataloader = get_lfw_dataloader(root_dir, pairs_file, batch_size)
    return evaluate_verification(model, dataloader, device)


def evaluate_cross_validation(model, dataloader, num_folds=10, device='cuda'):
    """
    Evaluate with k-fold cross-validation (for LFW).
    
    Args:
        model: Face recognition model
        dataloader: DataLoader with verification pairs
        num_folds: Number of folds
        device: Device to run on
    
    Returns:
        Dictionary with mean and std of metrics across folds
    """
    # Collect all data first
    all_data = []
    for batch in dataloader:
        all_data.append(batch)
    
    # Split into folds
    fold_size = len(all_data) // num_folds
    fold_results = []
    
    for fold in range(num_folds):
        # Create fold dataloader (simplified - in practice would properly split)
        fold_data = all_data[fold * fold_size:(fold + 1) * fold_size]
        
        # Evaluate fold
        fold_metrics = evaluate_verification(model, fold_data, device)
        fold_results.append(fold_metrics)
    
    # Aggregate results
    aggregated = {}
    for key in fold_results[0].keys():
        if isinstance(fold_results[0][key], dict):
            aggregated[key] = {}
            for subkey in fold_results[0][key].keys():
                values = [r[key][subkey] for r in fold_results]
                aggregated[key][subkey] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        else:
            values = [r[key] for r in fold_results]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    return aggregated
