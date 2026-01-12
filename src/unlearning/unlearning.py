"""
Core unlearning operations
Main interface for machine unlearning
"""

import torch
import numpy as np
from .identity_removal import IdentityUnlearner
from .linear_ops import (
    unlearn_orthogonal_projection,
    unlearn_svd_update,
    unlearn_linear_combination
)


class UnlearningEngine:
    """
    Main unlearning engine for linearized face recognition models.
    """
    
    def __init__(self, linearizer, method='orthogonal_projection', **kwargs):
        """
        Initialize unlearning engine.
        
        Args:
            linearizer: Linearizer instance
            method: Unlearning method
            **kwargs: Additional method-specific parameters
        """
        self.linearizer = linearizer
        self.method = method
        self.kwargs = kwargs
        self.unlearner = IdentityUnlearner(linearizer, method)
    
    def unlearn(self, dataloader, identity_ids_to_unlearn, device='cuda'):
        """
        Unlearn specific identities.
        
        Args:
            dataloader: DataLoader with training data
            identity_ids_to_unlearn: List of identity IDs to unlearn
            device: Device to run on
        
        Returns:
            Updated linear operator
        """
        return self.unlearner.unlearn_from_data(
            dataloader, identity_ids_to_unlearn, device
        )
    
    def unlearn_from_embeddings(self, embeddings, identity_labels, identity_ids_to_unlearn):
        """
        Unlearn from precomputed embeddings.
        
        Args:
            embeddings: Face embeddings [batch_size, embedding_size]
            identity_labels: Identity labels [batch_size]
            identity_ids_to_unlearn: List of identity IDs to unlearn
        
        Returns:
            Updated linear operator
        """
        return self.unlearner.unlearn_identities(
            embeddings, identity_labels, identity_ids_to_unlearn
        )
    
    def verify(self, embeddings, identity_labels, identity_ids_to_unlearn):
        """
        Verify unlearning effectiveness.
        
        Args:
            embeddings: Face embeddings [batch_size, embedding_size]
            identity_labels: Identity labels [batch_size]
            identity_ids_to_unlearn: List of unlearned identity IDs
        
        Returns:
            Verification results dictionary
        """
        return self.unlearner.verify_unlearning(
            embeddings, identity_labels, identity_ids_to_unlearn
        )
    
    def compare_with_retraining(self, dataloader, identity_ids_to_unlearn, 
                                retrained_model, device='cuda'):
        """
        Compare unlearning results with retrained model.
        
        Args:
            dataloader: DataLoader for evaluation
            identity_ids_to_unlearn: List of unlearned identity IDs
            retrained_model: Model retrained without these identities
            device: Device to run on
        
        Returns:
            Comparison metrics
        """
        self.linearizer.model.eval()
        retrained_model.eval()
        
        results = {
            'unlearned_model': [],
            'retrained_model': []
        }
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                
                # Get embeddings from unlearned model
                unlearned_embeddings = self.linearizer(images)
                
                # Get embeddings from retrained model
                retrained_embeddings = retrained_model.extract_features(images)
                
                results['unlearned_model'].append(unlearned_embeddings.cpu())
                results['retrained_model'].append(retrained_embeddings.cpu())
        
        # Stack embeddings
        results['unlearned_model'] = torch.cat(results['unlearned_model'], dim=0)
        results['retrained_model'] = torch.cat(results['retrained_model'], dim=0)
        
        # Compute similarity
        unlearned_norm = torch.nn.functional.normalize(
            results['unlearned_model'], p=2, dim=1
        )
        retrained_norm = torch.nn.functional.normalize(
            results['retrained_model'], p=2, dim=1
        )
        
        similarity = (unlearned_norm * retrained_norm).sum(dim=1).mean()
        results['similarity'] = float(similarity)
        
        return results
