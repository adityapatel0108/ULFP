"""
Linearization transformation functions
Convert neural networks to linear operators in learned latent space
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd


class LinearizedModel(nn.Module):
    """
    Linearized representation of a neural network in learned latent space.
    """
    
    def __init__(self, original_model, invertible_net, embedding_size=512):
        """
        Initialize linearized model.
        
        Args:
            original_model: Original face recognition model
            invertible_net: Invertible network for latent space
            embedding_size: Dimension of embeddings
        """
        super(LinearizedModel, self).__init__()
        self.original_model = original_model
        self.invertible_net = invertible_net
        self.embedding_size = embedding_size
        
        # Linear operator in latent space (learned)
        self.linear_op = nn.Linear(embedding_size, embedding_size, bias=False)
        
        # Initialize linear operator as identity
        nn.init.eye_(self.linear_op.weight)
    
    def forward(self, x):
        """
        Forward pass through linearized model.
        
        Args:
            x: Input images [batch_size, 3, 112, 112]
        
        Returns:
            Embeddings in original space [batch_size, embedding_size]
        """
        # Get embeddings from original model
        with torch.no_grad():
            embeddings = self.original_model.extract_features(x)
        
        # Transform to latent space
        latent = self.invertible_net(embeddings, reverse=False)
        
        # Apply linear operator
        latent_transformed = self.linear_op(latent)
        
        # Transform back to original space
        embeddings_transformed = self.invertible_net(latent_transformed, reverse=True)
        
        return embeddings_transformed
    
    def get_linear_operator(self):
        """Get the linear operator matrix."""
        return self.linear_op.weight.data.cpu().numpy()
    
    def set_linear_operator(self, matrix):
        """Set the linear operator matrix."""
        if isinstance(matrix, np.ndarray):
            matrix = torch.from_numpy(matrix).float()
        self.linear_op.weight.data = matrix.to(self.linear_op.weight.device)


def compute_linear_operator(model, dataloader, device='cuda', num_samples=1000):
    """
    Compute linear operator by analyzing model behavior on data.
    
    Args:
        model: Face recognition model
        dataloader: DataLoader for training data
        device: Device to run on
        num_samples: Number of samples to use
    
    Returns:
        Linear operator matrix [embedding_size, embedding_size]
    """
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        count = 0
        for images, _ in dataloader:
            if count >= num_samples:
                break
            
            images = images.to(device)
            embeddings = model.extract_features(images)
            embeddings_list.append(embeddings.cpu())
            count += len(images)
    
    # Stack all embeddings
    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    # Compute covariance matrix
    all_embeddings = all_embeddings.numpy()
    mean = np.mean(all_embeddings, axis=0, keepdims=True)
    centered = all_embeddings - mean
    
    # SVD to get principal directions
    U, s, Vt = svd(centered.T, full_matrices=False)
    
    # Linear operator approximates the transformation
    # This is a simplified version - full implementation would learn this
    linear_op = Vt.T @ np.diag(s) @ Vt
    
    return linear_op


def linearize_model(model, invertible_net, dataloader=None, device='cuda'):
    """
    Create linearized version of a model.
    
    Args:
        model: Original face recognition model
        invertible_net: Invertible network
        dataloader: Optional DataLoader for computing linear operator
        device: Device to run on
    
    Returns:
        LinearizedModel instance
    """
    embedding_size = model.embedding_size
    
    linearized = LinearizedModel(model, invertible_net, embedding_size)
    
    # If dataloader provided, compute initial linear operator
    if dataloader is not None:
        linear_op = compute_linear_operator(model, dataloader, device)
        linearized.set_linear_operator(linear_op)
    
    return linearized
