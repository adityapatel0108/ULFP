"""
Standard face recognition evaluation metrics
FNMR, FMR, EER, AUC
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch


def compute_cosine_similarity(embeddings1, embeddings2):
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings [n, embedding_size]
        embeddings2: Second set of embeddings [n, embedding_size]
    
    Returns:
        Similarity scores [n]
    """
    if isinstance(embeddings1, torch.Tensor):
        embeddings1 = embeddings1.cpu().numpy()
    if isinstance(embeddings2, torch.Tensor):
        embeddings2 = embeddings2.cpu().numpy()
    
    # Normalize
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarity = np.sum(embeddings1 * embeddings2, axis=1)
    
    return similarity


def compute_verification_accuracy(embeddings1, embeddings2, labels, threshold=None):
    """
    Compute face verification accuracy.
    
    Args:
        embeddings1: First set of embeddings [n, embedding_size]
        embeddings2: Second set of embeddings [n, embedding_size]
        labels: True labels (1 for same, 0 for different) [n]
        threshold: Similarity threshold (if None, optimal threshold is computed)
    
    Returns:
        Dictionary with accuracy, threshold, and predictions
    """
    # Compute similarities
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Find optimal threshold if not provided
    if threshold is None:
        # Use EER threshold
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr
        eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
        threshold = thresholds[eer_threshold_idx]
    
    # Predictions
    predictions = (similarities >= threshold).astype(int)
    
    # Accuracy
    accuracy = np.mean(predictions == labels)
    
    return {
        'accuracy': float(accuracy),
        'threshold': float(threshold),
        'predictions': predictions,
        'similarities': similarities
    }


def compute_fnmr_fmr(embeddings1, embeddings2, labels, fmr_threshold):
    """
    Compute False Non-Match Rate (FNMR) at a fixed False Match Rate (FMR).
    
    Args:
        embeddings1: First set of embeddings [n, embedding_size]
        embeddings2: Second set of embeddings [n, embedding_size]
        labels: True labels (1 for same, 0 for different) [n]
        fmr_threshold: Target FMR threshold
    
    Returns:
        Dictionary with FNMR, FMR, and threshold
    """
    # Compute similarities
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Separate genuine and impostor pairs
    genuine_mask = labels == 1
    impostor_mask = labels == 0
    
    genuine_similarities = similarities[genuine_mask]
    impostor_similarities = similarities[impostor_mask]
    
    if len(impostor_similarities) == 0 or len(genuine_similarities) == 0:
        return {
            'fnmr': 1.0,
            'fmr': 1.0,
            'threshold': 0.0
        }
    
    # Find threshold that gives target FMR
    impostor_sorted = np.sort(impostor_similarities)
    threshold_idx = int((1 - fmr_threshold) * len(impostor_sorted))
    threshold = impostor_sorted[threshold_idx] if threshold_idx < len(impostor_sorted) else impostor_sorted[-1]
    
    # Compute FNMR at this threshold
    fnmr = np.mean(genuine_similarities < threshold)
    
    # Actual FMR
    fmr = np.mean(impostor_similarities >= threshold)
    
    return {
        'fnmr': float(fnmr),
        'fmr': float(fmr),
        'threshold': float(threshold)
    }


def compute_eer(embeddings1, embeddings2, labels):
    """
    Compute Equal Error Rate (EER).
    
    Args:
        embeddings1: First set of embeddings [n, embedding_size]
        embeddings2: Second set of embeddings [n, embedding_size]
        labels: True labels (1 for same, 0 for different) [n]
    
    Returns:
        Dictionary with EER and threshold
    """
    # Compute similarities
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    fnr = 1 - tpr
    
    # Find EER (where FPR = FNR)
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    return {
        'eer': float(eer),
        'threshold': float(eer_threshold)
    }


def compute_auc(embeddings1, embeddings2, labels):
    """
    Compute Area Under ROC Curve (AUC).
    
    Args:
        embeddings1: First set of embeddings [n, embedding_size]
        embeddings2: Second set of embeddings [n, embedding_size]
        labels: True labels (1 for same, 0 for different) [n]
    
    Returns:
        AUC score
    """
    # Compute similarities
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Compute AUC
    try:
        auc_score = roc_auc_score(labels, similarities)
    except ValueError:
        # If only one class present
        auc_score = 0.5
    
    return float(auc_score)


def compute_all_metrics(embeddings1, embeddings2, labels, fmr_thresholds=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
    """
    Compute all standard face recognition metrics.
    
    Args:
        embeddings1: First set of embeddings [n, embedding_size]
        embeddings2: Second set of embeddings [n, embedding_size]
        labels: True labels (1 for same, 0 for different) [n]
        fmr_thresholds: List of FMR thresholds for FNMR computation
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Verification accuracy
    acc_result = compute_verification_accuracy(embeddings1, embeddings2, labels)
    metrics['accuracy'] = acc_result['accuracy']
    metrics['threshold'] = acc_result['threshold']
    
    # EER
    eer_result = compute_eer(embeddings1, embeddings2, labels)
    metrics['eer'] = eer_result['eer']
    metrics['eer_threshold'] = eer_result['threshold']
    
    # AUC
    metrics['auc'] = compute_auc(embeddings1, embeddings2, labels)
    
    # FNMR at different FMR thresholds
    metrics['fnmr_at_fmr'] = {}
    for fmr_threshold in fmr_thresholds:
        fnmr_result = compute_fnmr_fmr(embeddings1, embeddings2, labels, fmr_threshold)
        metrics['fnmr_at_fmr'][f'fnmr@fmr={fmr_threshold:.0e}'] = fnmr_result['fnmr']
    
    return metrics
