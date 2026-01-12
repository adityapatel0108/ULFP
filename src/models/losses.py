"""
Loss functions for face recognition training
ArcFace, MagFace, AdaFace implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFace(nn.Module):
    """
    ArcFace loss function.
    
    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    
    def __init__(self, embedding_size=512, num_classes=85000, margin=0.5, scale=64.0):
        """
        Initialize ArcFace loss.
        
        Args:
            embedding_size: Dimension of face embeddings
            num_classes: Number of identities
            margin: Angular margin (in radians)
            scale: Feature scale
        """
        super(ArcFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, labels):
        """
        Forward pass.
        
        Args:
            embeddings: Face embeddings [batch_size, embedding_size]
            labels: Identity labels [batch_size]
        
        Returns:
            Loss value
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Add angular margin
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Scale and apply margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


class MagFace(nn.Module):
    """
    MagFace loss function.
    
    Reference: Meng et al., "MagFace: A Universal Representation for Face Recognition"
    """
    
    def __init__(self, embedding_size=512, num_classes=85000, margin=0.5, scale=64.0):
        super(MagFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        """Forward pass for MagFace."""
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # MagFace uses magnitude-aware margin
        # Simplified version - full implementation would include magnitude terms
        output = cosine * self.scale
        
        return output


class AdaFace(nn.Module):
    """
    AdaFace loss function.
    
    Reference: Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition"
    """
    
    def __init__(self, embedding_size=512, num_classes=85000, margin=0.4, scale=64.0):
        super(AdaFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        """Forward pass for AdaFace."""
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # AdaFace uses adaptive margin based on image quality
        # Simplified version
        output = cosine * self.scale
        
        return output
