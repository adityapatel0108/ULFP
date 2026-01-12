"""
Identity-specific unlearning operations
"""

import torch
import numpy as np
from .linear_ops import compute_unlearning_update
from ..linearizer.utils import compute_identity_directions


class IdentityUnlearner:
    """
    Class for unlearning specific identities from a linearized model.
    """
    
    def __init__(self, linearizer, method='orthogonal_projection'):
        """
        Initialize identity unlearner.
        
        Args:
            linearizer: Linearizer instance
            method: Unlearning method ('orthogonal_projection', 'svd_update', 'linear_combination')
        """
        self.linearizer = linearizer
        self.method = method
    
    def unlearn_identities(self, embeddings, identity_labels, identity_ids_to_unlearn):
        """
        Unlearn specific identities from the linearized model.
        
        Args:
            embeddings: Face embeddings [batch_size, embedding_size]
            identity_labels: Identity labels [batch_size]
            identity_ids_to_unlearn: List of identity IDs to unlearn
        
        Returns:
            Updated linear operator
        """
        # Get current linear operator
        linear_operator = self.linearizer.get_linear_operator()
        
        # Compute unlearning update
        updated_operator = compute_unlearning_update(
            linear_operator=linear_operator,
            embeddings=embeddings,
            identity_labels=identity_labels,
            identity_ids_to_unlearn=identity_ids_to_unlearn,
            method=self.method
        )
        
        # Update linearizer
        self.linearizer.set_linear_operator(updated_operator)
        
        return updated_operator
    
    def unlearn_from_data(self, dataloader, identity_ids_to_unlearn, device='cuda'):
        """
        Unlearn identities from a DataLoader.
        
        Args:
            dataloader: DataLoader with (images, labels)
            identity_ids_to_unlearn: List of identity IDs to unlearn
            device: Device to run on
        
        Returns:
            Updated linear operator
        """
        self.linearizer.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                embeddings = self.linearizer.model.extract_features(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all embeddings and labels
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        # Perform unlearning
        updated_operator = self.unlearn_identities(
            embeddings, labels, identity_ids_to_unlearn
        )
        
        return updated_operator
    
    def get_identity_directions(self, embeddings, identity_labels):
        """
        Get principal directions for each identity.
        
        Args:
            embeddings: Face embeddings [batch_size, embedding_size]
            identity_labels: Identity labels [batch_size]
        
        Returns:
            Dictionary mapping identity_id to directions
        """
        return compute_identity_directions(embeddings, identity_labels)
    
    def verify_unlearning(self, embeddings, identity_labels, identity_ids_to_unlearn):
        """
        Verify that identities have been unlearned.
        
        Args:
            embeddings: Face embeddings [batch_size, embedding_size]
            identity_labels: Identity labels [batch_size]
            identity_ids_to_unlearn: List of identity IDs that should be unlearned
        
        Returns:
            Dictionary with verification metrics
        """
        # Transform embeddings to latent space
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        
        latent_embeddings = self.linearizer.to_latent_space(embeddings)
        
        # Apply linear operator
        transformed_latent = self.linearizer.apply_linear_operator(latent_embeddings)
        
        # Transform back
        transformed_embeddings = self.linearizer.from_latent_space(transformed_latent)
        
        # Compute identity directions before and after
        original_directions = compute_identity_directions(
            embeddings.cpu().numpy(), identity_labels
        )
        transformed_directions = compute_identity_directions(
            transformed_embeddings.cpu().numpy(), identity_labels
        )
        
        # Compare directions for unlearned identities
        results = {}
        for identity_id in identity_ids_to_unlearn:
            if identity_id in original_directions and identity_id in transformed_directions:
                orig_dir = original_directions[identity_id]['directions']
                trans_dir = transformed_directions[identity_id]['directions']
                
                # Compute similarity (should be low if unlearning worked)
                similarity = np.abs(np.trace(orig_dir @ trans_dir.T))
                results[identity_id] = {
                    'similarity': float(similarity),
                    'unlearned': similarity < 0.1  # Threshold
                }
        
        return results
