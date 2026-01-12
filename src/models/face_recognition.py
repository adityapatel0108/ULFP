"""
Complete face recognition model wrapper
"""

import torch
import torch.nn as nn
from .losses import ArcFace, MagFace, AdaFace


class FaceRecognitionModel(nn.Module):
    """
    Complete face recognition model with backbone and optional loss.
    """
    
    def __init__(self, backbone, embedding_size=512, loss_type='arcface', 
                 num_classes=None, margin=0.5, scale=64.0):
        """
        Initialize face recognition model.
        
        Args:
            backbone: Backbone network (e.g., iResNet)
            embedding_size: Dimension of face embeddings
            loss_type: Type of loss ('arcface', 'magface', 'adaface', None)
            num_classes: Number of identities (required if loss_type is not None)
            margin: Margin for loss function
            scale: Scale for loss function
        """
        super(FaceRecognitionModel, self).__init__()
        self.backbone = backbone
        self.embedding_size = embedding_size
        self.loss_type = loss_type
        
        # Initialize loss function if specified
        if loss_type == 'arcface' and num_classes is not None:
            self.loss_fn = ArcFace(embedding_size, num_classes, margin, scale)
        elif loss_type == 'magface' and num_classes is not None:
            self.loss_fn = MagFace(embedding_size, num_classes, margin, scale)
        elif loss_type == 'adaface' and num_classes is not None:
            self.loss_fn = AdaFace(embedding_size, num_classes, margin, scale)
        else:
            self.loss_fn = None
    
    def forward(self, x, labels=None):
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, 112, 112]
            labels: Optional identity labels [batch_size]
        
        Returns:
            If labels provided: logits for loss computation
            Otherwise: face embeddings [batch_size, embedding_size]
        """
        # Extract features
        embeddings = self.backbone(x)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # If labels provided and loss function exists, compute logits
        if labels is not None and self.loss_fn is not None:
            logits = self.loss_fn(embeddings, labels)
            return logits
        
        return embeddings
    
    def extract_features(self, x):
        """
        Extract face embeddings without computing loss.
        
        Args:
            x: Input images [batch_size, 3, 112, 112]
        
        Returns:
            Face embeddings [batch_size, embedding_size]
        """
        return self.forward(x, labels=None)
    
    def get_backbone(self):
        """Get the backbone network."""
        return self.backbone
