"""
Linear algebra utilities for Linearizer framework
SVD, projections, and other linear operations
"""

import torch
import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.linalg import null_space


def svd_decomposition(matrix, k=None):
    """
    Perform SVD decomposition.
    
    Args:
        matrix: Input matrix [m, n]
        k: Number of components to keep (None for full SVD)
    
    Returns:
        U, s, Vt: SVD components
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    
    U, s, Vt = scipy_svd(matrix, full_matrices=False)
    
    if k is not None and k < len(s):
        U = U[:, :k]
        s = s[:k]
        Vt = Vt[:k, :]
    
    return U, s, Vt


def orthogonal_projection(matrix, directions):
    """
    Project matrix onto orthogonal complement of given directions.
    
    Args:
        matrix: Input matrix [m, n]
        directions: Directions to remove [k, n]
    
    Returns:
        Projected matrix [m, n]
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    if isinstance(directions, torch.Tensor):
        directions = directions.cpu().numpy()
    
    # Normalize directions
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Compute projection matrix
    P = np.eye(matrix.shape[1]) - directions.T @ directions
    
    # Apply projection
    projected = matrix @ P
    
    return projected


def remove_subspace(matrix, subspace_basis):
    """
    Remove a subspace from a matrix using orthogonal projection.
    
    Args:
        matrix: Input matrix [m, n]
        subspace_basis: Basis vectors of subspace to remove [k, n]
    
    Returns:
        Matrix with subspace removed [m, n]
    """
    return orthogonal_projection(matrix, subspace_basis)


def low_rank_update(matrix, U, s, Vt, alpha=1.0):
    """
    Perform low-rank update: matrix + alpha * U @ diag(s) @ Vt.
    
    Args:
        matrix: Base matrix [m, n]
        U: Left singular vectors [m, k]
        s: Singular values [k]
        Vt: Right singular vectors [k, n]
        alpha: Update strength
    
    Returns:
        Updated matrix [m, n]
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    if isinstance(U, torch.Tensor):
        U = U.cpu().numpy()
    if isinstance(s, torch.Tensor):
        s = s.cpu().numpy()
    if isinstance(Vt, torch.Tensor):
        Vt = Vt.cpu().numpy()
    
    update = alpha * U @ np.diag(s) @ Vt
    updated = matrix + update
    
    return updated


def compute_identity_directions(embeddings, identity_labels):
    """
    Compute principal directions for each identity.
    
    Args:
        embeddings: Face embeddings [batch_size, embedding_size]
        identity_labels: Identity labels [batch_size]
    
    Returns:
        Dictionary mapping identity_id to principal directions
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(identity_labels, torch.Tensor):
        identity_labels = identity_labels.cpu().numpy()
    
    identity_directions = {}
    unique_identities = np.unique(identity_labels)
    
    for identity_id in unique_identities:
        # Get embeddings for this identity
        mask = identity_labels == identity_id
        identity_embeddings = embeddings[mask]
        
        if len(identity_embeddings) < 2:
            continue
        
        # Compute mean
        mean = np.mean(identity_embeddings, axis=0)
        centered = identity_embeddings - mean
        
        # SVD to get principal directions
        if centered.shape[0] > 1:
            U, s, Vt = svd_decomposition(centered.T, k=min(10, centered.shape[0]-1))
            # Store top directions
            identity_directions[identity_id] = {
                'mean': mean,
                'directions': Vt[:min(5, len(s)), :],  # Top 5 directions
                'singular_values': s[:min(5, len(s))]
            }
    
    return identity_directions


def compute_linear_combination(operators, weights):
    """
    Compute linear combination of linear operators.
    
    Args:
        operators: List of linear operators [n, embedding_size, embedding_size]
        weights: Combination weights [n]
    
    Returns:
        Combined operator [embedding_size, embedding_size]
    """
    if isinstance(operators, list):
        operators = np.array(operators)
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Compute weighted combination
    combined = np.zeros_like(operators[0])
    for op, w in zip(operators, weights):
        combined += w * op
    
    return combined
