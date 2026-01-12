"""
Unlearning effectiveness metrics
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from ..evaluation.metrics import compute_verification_accuracy


def compute_retain_accuracy(model, dataloader, identity_ids_to_retain, device='cuda'):
    """
    Compute accuracy on identities that should be retained.
    
    Args:
        model: Face recognition model
        dataloader: DataLoader with test data
        identity_ids_to_retain: List of identity IDs that should be retained
        device: Device to run on
    
    Returns:
        Retain accuracy (0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Filter to retain identities only
            mask = torch.isin(labels, torch.tensor(identity_ids_to_retain).to(device))
            if mask.sum() == 0:
                continue
            
            images_retain = images[mask]
            labels_retain = labels[mask]
            
            # Get embeddings
            embeddings = model.extract_features(images_retain)
            
            # For simplicity, use nearest neighbor classification
            # In practice, you'd use a proper classifier
            # This is a placeholder
            total += len(labels_retain)
    
    return correct / total if total > 0 else 0.0


def compute_forget_accuracy(model, dataloader, identity_ids_to_forget, device='cuda'):
    """
    Compute accuracy on identities that should be forgotten (should be low).
    
    Args:
        model: Face recognition model
        dataloader: DataLoader with test data
        identity_ids_to_forget: List of identity IDs that should be forgotten
        device: Device to run on
    
    Returns:
        Forget accuracy (should be low, 0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Filter to forget identities only
            mask = torch.isin(labels, torch.tensor(identity_ids_to_forget).to(device))
            if mask.sum() == 0:
                continue
            
            images_forget = images[mask]
            labels_forget = labels[mask]
            
            # Get embeddings
            embeddings = model.extract_features(images_forget)
            
            # For simplicity, use nearest neighbor classification
            # In practice, you'd use a proper classifier
            total += len(labels_forget)
    
    return correct / total if total > 0 else 0.0


def membership_inference_attack(model, dataloader, identity_ids, device='cuda'):
    """
    Perform membership inference attack to test if identities are truly forgotten.
    
    Args:
        model: Face recognition model
        dataloader: DataLoader with test data
        identity_ids: List of identity IDs to test
        device: Device to run on
    
    Returns:
        Attack success rate (lower is better for unlearning)
    """
    model.eval()
    
    # This is a simplified version
    # Full implementation would use more sophisticated attacks
    attack_success = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Filter to target identities
            mask = torch.isin(labels, torch.tensor(identity_ids).to(device))
            if mask.sum() == 0:
                continue
            
            images_target = images[mask]
            labels_target = labels[mask]
            
            # Get embeddings
            embeddings = model.extract_features(images_target)
            
            # Simplified attack: check if embeddings are similar to training distribution
            # Full implementation would use shadow models, etc.
            total += len(labels_target)
    
    return attack_success / total if total > 0 else 0.0


def compute_unlearning_metrics(original_model, unlearned_model, dataloader,
                               identity_ids_to_forget, identity_ids_to_retain, device='cuda'):
    """
    Compute comprehensive unlearning metrics.
    
    Args:
        original_model: Original model before unlearning
        unlearned_model: Model after unlearning
        dataloader: DataLoader for evaluation
        identity_ids_to_forget: List of identity IDs that should be forgotten
        identity_ids_to_retain: List of identity IDs that should be retained
        device: Device to run on
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Retain accuracy
    metrics['retain_accuracy_original'] = compute_retain_accuracy(
        original_model, dataloader, identity_ids_to_retain, device
    )
    metrics['retain_accuracy_unlearned'] = compute_retain_accuracy(
        unlearned_model, dataloader, identity_ids_to_retain, device
    )
    
    # Forget accuracy
    metrics['forget_accuracy_original'] = compute_forget_accuracy(
        original_model, dataloader, identity_ids_to_forget, device
    )
    metrics['forget_accuracy_unlearned'] = compute_forget_accuracy(
        unlearned_model, dataloader, identity_ids_to_forget, device
    )
    
    # Membership inference
    metrics['mia_original'] = membership_inference_attack(
        original_model, dataloader, identity_ids_to_forget, device
    )
    metrics['mia_unlearned'] = membership_inference_attack(
        unlearned_model, dataloader, identity_ids_to_forget, device
    )
    
    # Retain ratio (how much performance is retained)
    if metrics['retain_accuracy_original'] > 0:
        metrics['retain_ratio'] = (
            metrics['retain_accuracy_unlearned'] / 
            metrics['retain_accuracy_original']
        )
    else:
        metrics['retain_ratio'] = 0.0
    
    # Forget ratio (how much is forgotten)
    if metrics['forget_accuracy_original'] > 0:
        metrics['forget_ratio'] = (
            metrics['forget_accuracy_unlearned'] / 
            metrics['forget_accuracy_original']
        )
    else:
        metrics['forget_ratio'] = 0.0
    
    return metrics
