"""
Core Linearizer framework
Main interface for linearizing face recognition models
Implements the sandwich architecture: f(x) = g⁻¹ᵧ(Agₓ(x))
Based on Berman et al. "Who Said Neural Networks Aren't Linear?" (2025)
"""

import torch
import torch.nn as nn
from .invertible_net import ImageToLatentNetwork, LatentToEmbeddingNetwork
from .linearization import LinearizedModel, linearize_model


class Linearizer(nn.Module):
    """
    Main Linearizer class implementing the sandwich architecture.
    
    The sandwich architecture: f(x) = g⁻¹ᵧ(Agₓ(x))
    Where:
    - gₓ: ImageToLatentNetwork (maps images to latent space)
    - A: Linear operator in latent space
    - g⁻¹ᵧ: LatentToEmbeddingNetwork (maps latent space to embeddings)
    """
    
    def __init__(self, model, embedding_size=512, latent_dim=512,
                 num_blocks=4, hidden_dim=1024, num_layers=3,
                 image_size=(112, 112)):
        """
        Initialize Linearizer with sandwich architecture.
        
        Args:
            model: Face recognition model to linearize (for reference embeddings)
            embedding_size: Dimension of face embeddings
            latent_dim: Dimension of latent space
            num_blocks: Number of invertible blocks in each network
            hidden_dim: Hidden dimension for invertible networks
            num_layers: Number of layers in coupling networks
            image_size: Input image size (height, width)
        """
        super(Linearizer, self).__init__()
        self.model = model
        self.embedding_size = embedding_size
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Create gₓ: Image -> Latent
        self.g_x = ImageToLatentNetwork(
            image_size=image_size,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Create g⁻¹ᵧ: Latent -> Embedding
        self.g_y_inv = LatentToEmbeddingNetwork(
            latent_dim=latent_dim,
            embedding_dim=embedding_size,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Linear operator A in latent space
        self.linear_op = nn.Linear(latent_dim, latent_dim, bias=False)
        # Initialize as identity
        nn.init.eye_(self.linear_op.weight)
    
    def forward(self, x):
        """
        Forward pass through linearized model using sandwich architecture.
        f(x) = g⁻¹ᵧ(Agₓ(x))
        
        Args:
            x: Input images [batch_size, 3, height, width]
        
        Returns:
            Embeddings [batch_size, embedding_size]
        """
        # Step 1: gₓ(x) - Map images to latent space
        z = self.g_x(x, reverse=False)
        
        # Step 2: A(z) - Apply linear operator in latent space
        z_transformed = self.linear_op(z)
        
        # Step 3: g⁻¹ᵧ(z_transformed) - Map latent space to embeddings
        embeddings = self.g_y_inv(z_transformed, reverse=False)
        
        return embeddings
    
    def get_linear_operator(self):
        """Get the linear operator matrix in latent space."""
        return self.linear_op.weight.data.cpu().numpy()
    
    def set_linear_operator(self, matrix):
        """Set the linear operator matrix in latent space."""
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy()
        matrix = torch.from_numpy(matrix).float()
        self.linear_op.weight.data = matrix.to(self.linear_op.weight.device)
    
    def to_latent_space(self, x):
        """
        Transform images to latent space using gₓ.
        
        Args:
            x: Input images [batch_size, 3, height, width]
        
        Returns:
            Latent vectors [batch_size, latent_dim]
        """
        return self.g_x(x, reverse=False)
    
    def from_latent_space(self, z):
        """
        Transform latent vectors to embeddings using g⁻¹ᵧ.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            Embeddings [batch_size, embedding_size]
        """
        return self.g_y_inv(z, reverse=False)
    
    def apply_linear_operator(self, z):
        """
        Apply linear operator to latent vectors.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            Transformed latent vectors [batch_size, latent_dim]
        """
        return self.linear_op(z)
    
    def train_linearizer(self, dataloader, num_epochs=100, lr=0.0001, device='cuda'):
        """
        Train the invertible networks and linear operator.
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of training epochs
            lr: Learning rate
            device: Device to run on
        """
        self.train()
        
        # Collect all trainable parameters
        trainable_params = (
            list(self.g_x.parameters()) + 
            list(self.g_y_inv.parameters()) + 
            list(self.linear_op.parameters())
        )
        
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        criterion = nn.MSELoss()
        
        print(f"Training Linearizer for {num_epochs} epochs...")
        print(f"Total parameters: {sum(p.numel() for p in trainable_params):,}")
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for images, labels in dataloader:
                images = images.to(device)
                
                # Get original embeddings from reference model
                with torch.no_grad():
                    original_embeddings = self.model.extract_features(images)
                
                # Get linearized embeddings using sandwich architecture
                linearized_embeddings = self.forward(images)
                
                # Compute reconstruction loss
                loss = criterion(linearized_embeddings, original_embeddings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        self.eval()
        print("Training completed!")