"""
Unlearning-specific evaluation metrics
"""

import torch
import numpy as np
from .metrics import compute_all_metrics
from ..unlearning.evaluation import (
    compute_retain_accuracy,
    compute_forget_accuracy,
    membership_inference_attack,
    compute_unlearning_metrics
)


def evaluate_unlearning_effectiveness(original_model, unlearned_model, 
                                     dataloader, identity_ids_to_forget,
                                     identity_ids_to_retain, device='cuda'):
    """
    Comprehensive evaluation of unlearning effectiveness.
    
    Args:
        original_model: Model before unlearning
        unlearned_model: Model after unlearning
        dataloader: DataLoader for evaluation
        identity_ids_to_forget: List of identity IDs that should be forgotten
        identity_ids_to_retain: List of identity IDs that should be retained
        device: Device to run on
    
    Returns:
        Dictionary with all unlearning metrics
    """
    return compute_unlearning_metrics(
        original_model, unlearned_model, dataloader,
        identity_ids_to_forget, identity_ids_to_retain, device
    )


def compute_forget_score(original_model, unlearned_model, dataloader,
                        identity_ids_to_forget, device='cuda'):
    """
    Compute forget score (from RAD paper).
    Measures how well identities are forgotten.
    
    Args:
        original_model: Model before unlearning
        unlearned_model: Model after unlearning
        dataloader: DataLoader with forget set
        identity_ids_to_forget: List of identity IDs to forget
        device: Device to run on
    
    Returns:
        Forget score (higher is better for unlearning)
    """
    original_model.eval()
    unlearned_model.eval()
    
    original_embeddings = []
    unlearned_embeddings = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Filter to forget identities
            mask = torch.isin(labels, torch.tensor(identity_ids_to_forget).to(device))
            if mask.sum() == 0:
                continue
            
            images = images[mask].to(device)
            
            # Get embeddings
            orig_emb = original_model.extract_features(images)
            unlearn_emb = unlearned_model.extract_features(images)
            
            original_embeddings.append(orig_emb.cpu())
            unlearned_embeddings.append(unlearn_emb.cpu())
    
    if len(original_embeddings) == 0:
        return 0.0
    
    # Stack embeddings
    original_embeddings = torch.cat(original_embeddings, dim=0)
    unlearned_embeddings = torch.cat(unlearned_embeddings, dim=0)
    
    # Normalize
    original_embeddings = torch.nn.functional.normalize(original_embeddings, p=2, dim=1)
    unlearned_embeddings = torch.nn.functional.normalize(unlearned_embeddings, p=2, dim=1)
    
    # Compute distance (how different are the embeddings)
    # Higher distance = better forgetting
    distance = torch.norm(original_embeddings - unlearned_embeddings, p=2, dim=1).mean()
    
    return float(distance.item())


def evaluate_verification_after_unlearning(model, verification_dataloader, device='cuda'):
    """
    Evaluate face verification performance after unlearning.
    
    Args:
        model: Unlearned model
        verification_dataloader: DataLoader with verification pairs
        device: Device to run on
    
    Returns:
        Verification metrics
    """
    from .verification import evaluate_verification
    return evaluate_verification(model, verification_dataloader, device)
