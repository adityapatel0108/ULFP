"""
Invertible neural network components for Linearizer framework
Based on Berman et al. "Who Said Neural Networks Aren't Linear?" (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InvertibleBlock(nn.Module):
    """
    Invertible block using coupling layers.
    Based on RealNVP/Glow architecture.
    """
    
    def __init__(self, dim, hidden_dim=512, num_layers=3):
        """
        Initialize invertible block.
        
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension for coupling networks
            num_layers: Number of layers in coupling networks
        """
        super(InvertibleBlock, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2
        
        # Coupling networks
        self.scale_net = self._make_network(self.half_dim, hidden_dim, num_layers, self.half_dim)
        self.translate_net = self._make_network(self.half_dim, hidden_dim, num_layers, self.half_dim)
    
    def _make_network(self, in_dim, hidden_dim, num_layers, out_dim):
        """Create a simple MLP network."""
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x, reverse=False):
        """
        Forward or inverse transform.
        
        Args:
            x: Input tensor [batch_size, dim]
            reverse: If True, perform inverse transform
        
        Returns:
            Transformed tensor
        """
        if not reverse:
            return self._forward(x)
        else:
            return self._inverse(x)
    
    def _forward(self, x):
        """Forward transform."""
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]
        
        # Compute scale and translation
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        
        # Apply affine coupling
        s = torch.tanh(s)  # Ensure invertibility
        y2 = x2 * torch.exp(s) + t
        y1 = x1
        
        return torch.cat([y1, y2], dim=1)
    
    def _inverse(self, y):
        """Inverse transform."""
        y1, y2 = y[:, :self.half_dim], y[:, self.half_dim:]
        
        # Compute scale and translation
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        
        # Apply inverse affine coupling
        s = torch.tanh(s)
        x2 = (y2 - t) * torch.exp(-s)
        x1 = y1
        
        return torch.cat([x1, x2], dim=1)


class InvertibleNetwork(nn.Module):
    """
    Stack of invertible blocks to create a learned latent space.
    """
    
    def __init__(self, dim, num_blocks=4, hidden_dim=1024, num_layers=3):
        """
        Initialize invertible network.
        
        Args:
            dim: Input/output dimension
            num_blocks: Number of invertible blocks
            hidden_dim: Hidden dimension for coupling networks
            num_layers: Number of layers in coupling networks
        """
        super(InvertibleNetwork, self).__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        
        # Create stack of invertible blocks
        self.blocks = nn.ModuleList([
            InvertibleBlock(dim, hidden_dim, num_layers)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x, reverse=False):
        """
        Forward or inverse transform through all blocks.
        
        Args:
            x: Input tensor [batch_size, dim]
            reverse: If True, perform inverse transform
        
        Returns:
            Transformed tensor
        """
        if not reverse:
            for block in self.blocks:
                x = block(x, reverse=False)
        else:
            for block in reversed(self.blocks):
                x = block(x, reverse=True)
        
        return x


class PermutationLayer(nn.Module):
    """
    Learnable permutation layer for invertible networks.
    """
    
    def __init__(self, dim):
        """
        Initialize permutation layer.
        
        Args:
            dim: Dimension of input
        """
        super(PermutationLayer, self).__init__()
        self.dim = dim
        # Use a learnable permutation matrix
        self.perm = nn.Parameter(torch.randn(dim, dim))
        # Initialize as identity-like
        with torch.no_grad():
            self.perm.data = torch.eye(dim) + 0.01 * torch.randn(dim, dim)
    
    def forward(self, x, reverse=False):
        """
        Apply permutation.
        
        Args:
            x: Input tensor [batch_size, dim]
            reverse: If True, apply inverse permutation
        
        Returns:
            Permuted tensor
        """
        if not reverse:
            return F.linear(x, self.perm)
        else:
            # Inverse permutation
            perm_inv = torch.inverse(self.perm)
            return F.linear(x, perm_inv)


class ImageToLatentNetwork(nn.Module):
    """
    Invertible network that maps images to latent space.
    Uses a CNN encoder followed by invertible blocks.
    Implements gₓ: X → Z (images to latent space)
    """
    
    def __init__(self, image_size=(112, 112), latent_dim=512, 
                 num_blocks=4, hidden_dim=1024, num_layers=3):
        """
        Initialize image-to-latent network.
        
        Args:
            image_size: Input image size (height, width)
            latent_dim: Dimension of latent space
            num_blocks: Number of invertible blocks
            hidden_dim: Hidden dimension for coupling networks
            num_layers: Number of layers in coupling networks
        """
        super(ImageToLatentNetwork, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # CNN encoder to map images to feature vectors
        # This is a learnable encoder that maps [B, 3, H, W] -> [B, latent_dim]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 112 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, latent_dim)
        )
        
        # Invertible network on the latent space
        self.invertible_net = InvertibleNetwork(
            dim=latent_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    
    def forward(self, x, reverse=False):
        """
        Forward or inverse transform.
        
        Args:
            x: Input images [batch_size, 3, height, width] if not reverse
               or latent vectors [batch_size, latent_dim] if reverse
            reverse: If True, perform inverse transform (latent -> image features)
        
        Returns:
            Latent vectors [batch_size, latent_dim] if not reverse
            Image features [batch_size, latent_dim] if reverse
        """
        if not reverse:
            # Image -> Latent: gₓ(x)
            # Encode image to feature vector
            features = self.encoder(x)
            # Apply invertible transformation
            latent = self.invertible_net(features, reverse=False)
            return latent
        else:
            # Latent -> Image features: gₓ⁻¹(z)
            # Apply inverse invertible transformation
            features = self.invertible_net(x, reverse=True)
            return features
    
    def inverse(self, z):
        """
        Inverse transform: latent -> image features.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            Image features [batch_size, latent_dim]
        """
        return self.forward(z, reverse=True)


class LatentToEmbeddingNetwork(nn.Module):
    """
    Invertible network that maps latent space to embedding space.
    Implements g⁻¹ᵧ: Z → Y (latent space to embeddings)
    """
    
    def __init__(self, latent_dim=512, embedding_dim=512,
                 num_blocks=4, hidden_dim=1024, num_layers=3):
        """
        Initialize latent-to-embedding network.
        
        Args:
            latent_dim: Dimension of latent space
            embedding_dim: Dimension of embedding space
            num_blocks: Number of invertible blocks
            hidden_dim: Hidden dimension for coupling networks
            num_layers: Number of layers in coupling networks
        """
        super(LatentToEmbeddingNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # If dimensions match, use invertible network directly
        if latent_dim == embedding_dim:
            self.invertible_net = InvertibleNetwork(
                dim=latent_dim,
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
            self.projection = None
        else:
            # Need projection layer if dimensions don't match
            self.invertible_net = InvertibleNetwork(
                dim=min(latent_dim, embedding_dim),
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
            if latent_dim > embedding_dim:
                self.projection = nn.Linear(latent_dim, embedding_dim)
            else:
                self.projection = nn.Linear(embedding_dim, latent_dim)
    
    def forward(self, z, reverse=False):
        """
        Forward or inverse transform.
        
        Args:
            z: Latent vectors [batch_size, latent_dim] if not reverse
               or embeddings [batch_size, embedding_dim] if reverse
            reverse: If True, perform inverse transform (embedding -> latent)
        
        Returns:
            Embeddings [batch_size, embedding_dim] if not reverse
            Latent vectors [batch_size, latent_dim] if reverse
        """
        if not reverse:
            # Latent -> Embedding: g⁻¹ᵧ(z)
            if self.projection is not None:
                if self.latent_dim > self.embedding_dim:
                    z = self.projection(z)
                else:
                    z = self.projection(z)
            
            # Apply invertible transformation
            embeddings = self.invertible_net(z, reverse=False)
            return embeddings
        else:
            # Embedding -> Latent: gᵧ(y)
            # Apply inverse invertible transformation
            z = self.invertible_net(z, reverse=True)
            
            if self.projection is not None:
                if self.latent_dim > self.embedding_dim:
                    # Need to expand dimension
                    z = F.pad(z, (0, self.latent_dim - self.embedding_dim))
                else:
                    # Need to reduce dimension
                    z = z[:, :self.latent_dim]
            
            return z
    
    def inverse(self, y):
        """
        Inverse transform: embedding -> latent.
        
        Args:
            y: Embeddings [batch_size, embedding_dim]
        
        Returns:
            Latent vectors [batch_size, latent_dim]
        """
        return self.forward(y, reverse=True)
