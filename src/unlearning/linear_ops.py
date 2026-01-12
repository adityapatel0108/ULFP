"""
Linear algebra operations for unlearning
"""

import torch
import numpy as np
from ..linearizer.utils import (
    svd_decomposition, 
    orthogonal_projection, 
    remove_subspace,
    low_rank_update,
    compute_identity_directions
)


def unlearn_orthogonal_projection(linear_operator, identity_directions, identity_ids):
    """
    Unlearn identities using orthogonal projection.
    
    Args:
        linear_operator: Linear operator matrix [embedding_size, embedding_size]
        identity_directions: Dictionary of identity directions (from compute_identity_directions)
        identity_ids: List of identity IDs to unlearn
    
    Returns:
        Updated linear operator [embedding_size, embedding_size]
    """
    if isinstance(linear_operator, torch.Tensor):
        linear_operator = linear_operator.cpu().numpy()
    
    # Collect all directions to remove
    directions_to_remove = []
    for identity_id in identity_ids:
        if identity_id in identity_directions:
            directions = identity_directions[identity_id]['directions']
            directions_to_remove.append(directions)
    
    if len(directions_to_remove) == 0:
        return linear_operator
    
    # Stack all directions
    all_directions = np.vstack(directions_to_remove)
    
    # Remove subspace using orthogonal projection
    updated_operator = remove_subspace(linear_operator, all_directions)
    
    return updated_operator


def unlearn_svd_update(linear_operator, identity_directions, identity_ids, rank=50, alpha=-0.1):
    """
    Unlearn identities using SVD-based low-rank update.
    
    Args:
        linear_operator: Linear operator matrix [embedding_size, embedding_size]
        identity_directions: Dictionary of identity directions
        identity_ids: List of identity IDs to unlearn
        rank: Rank of update
        alpha: Update strength (negative to remove)
    
    Returns:
        Updated linear operator [embedding_size, embedding_size]
    """
    if isinstance(linear_operator, torch.Tensor):
        linear_operator = linear_operator.cpu().numpy()
    
    # Collect all directions to remove
    directions_to_remove = []
    singular_values = []
    
    for identity_id in identity_ids:
        if identity_id in identity_directions:
            info = identity_directions[identity_id]
            directions = info['directions']
            svals = info['singular_values']
            directions_to_remove.append(directions)
            singular_values.append(svals)
    
    if len(directions_to_remove) == 0:
        return linear_operator
    
    # Stack all directions
    all_directions = np.vstack(directions_to_remove)
    all_svals = np.concatenate(singular_values)
    
    # Perform SVD on directions
    U, s, Vt = svd_decomposition(all_directions.T, k=min(rank, len(all_svals)))
    
    # Create low-rank update to remove these directions
    # Use negative alpha to subtract
    updated_operator = low_rank_update(linear_operator, U, s, Vt, alpha=alpha)
    
    return updated_operator


def unlearn_linear_combination(linear_operator, identity_directions, identity_ids, 
                               retain_operator=None, weight=0.1):
    """
    Unlearn identities using linear combination with a "clean" operator.
    
    Args:
        linear_operator: Current linear operator [embedding_size, embedding_size]
        identity_directions: Dictionary of identity directions
        identity_ids: List of identity IDs to unlearn
        retain_operator: Operator trained without these identities (if available)
        weight: Weight for the clean operator
    
    Returns:
        Updated linear operator [embedding_size, embedding_size]
    """
    if isinstance(linear_operator, torch.Tensor):
        linear_operator = linear_operator.cpu().numpy()
    
    if retain_operator is None:
        # If no clean operator, use orthogonal projection as fallback
        return unlearn_orthogonal_projection(linear_operator, identity_directions, identity_ids)
    
    if isinstance(retain_operator, torch.Tensor):
        retain_operator = retain_operator.cpu().numpy()
    
    # Linear combination: (1 - weight) * current + weight * clean
    updated_operator = (1 - weight) * linear_operator + weight * retain_operator
    
    return updated_operator


def compute_unlearning_update(linear_operator, embeddings, identity_labels, 
                             identity_ids_to_unlearn, method='orthogonal_projection'):
    """
    Compute unlearning update for specified identities.
    
    Args:
        linear_operator: Current linear operator [embedding_size, embedding_size]
        embeddings: Face embeddings [batch_size, embedding_size]
        identity_labels: Identity labels [batch_size]
        identity_ids_to_unlearn: List of identity IDs to unlearn
        method: Unlearning method ('orthogonal_projection', 'svd_update', 'linear_combination')
    
    Returns:
        Updated linear operator [embedding_size, embedding_size]
    """
    # Compute identity directions
    identity_directions = compute_identity_directions(embeddings, identity_labels)
    
    # Apply unlearning method
    if method == 'orthogonal_projection':
        updated_operator = unlearn_orthogonal_projection(
            linear_operator, identity_directions, identity_ids_to_unlearn
        )
    elif method == 'svd_update':
        updated_operator = unlearn_svd_update(
            linear_operator, identity_directions, identity_ids_to_unlearn
        )
    elif method == 'linear_combination':
        updated_operator = unlearn_linear_combination(
            linear_operator, identity_directions, identity_ids_to_unlearn
        )
    else:
        raise ValueError(f"Unknown unlearning method: {method}")
    
    return updated_operator
