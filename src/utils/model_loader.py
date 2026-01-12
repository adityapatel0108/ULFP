"""
Utilities to load pretrained face recognition models
Supports InsightFace models including buffalo_l
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Install with: pip install insightface")

from ..models.backbone import iresnet18, iresnet50, iresnet100
from ..models.face_recognition import FaceRecognitionModel


def load_insightface_model(model_name='buffalo_l', root_dir='~/.insightface'):
    """
    Load pretrained model from InsightFace using buffalo_l or other model packs.
    
    Args:
        model_name: Model name ('buffalo_l', 'buffalo_m', 'buffalo_s', etc.)
        root_dir: Root directory for InsightFace models
    
    Returns:
        FaceAnalysis app instance
    """
    if not INSIGHTFACE_AVAILABLE:
        raise ImportError("InsightFace is not installed. Install with: pip install insightface")
    
    # Initialize FaceAnalysis app
    app = FaceAnalysis(name=model_name, root=root_dir)
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    return app


def extract_model_from_insightface(app):
    """
    Extract the face recognition model from InsightFace FaceAnalysis app.
    
    Args:
        app: FaceAnalysis app instance
    
    Returns:
        PyTorch model (backbone + embedding layer)
    """
    # Create a wrapper that uses InsightFace's embedding function
    class InsightFaceWrapper(nn.Module):
        def __init__(self, app):
            super().__init__()
            self.app = app
            # Get embedding size from the model
            if hasattr(app, 'models') and 'recognition' in app.models:
                rec_model = app.models['recognition']
                # Try to get embedding size
                if hasattr(rec_model, 'output_layer'):
                    self.embedding_size = rec_model.output_layer.weight.shape[0]
                else:
                    self.embedding_size = 512  # Default for buffalo_l
            else:
                self.embedding_size = 512  # Default for buffalo_l
        
        def forward(self, x):
            """
            Forward pass through InsightFace model.
            
            Args:
                x: Input images [batch_size, 3, 112, 112] as torch.Tensor
            
            Returns:
                Face embeddings [batch_size, embedding_size]
            """
            # Convert tensor to numpy
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            
            # InsightFace expects BGR format and [0, 255] range
            # Our input is normalized to [-1, 1], so we need to convert
            batch_size = x_np.shape[0]
            embeddings_list = []
            
            # Try to get recognition model directly for faster processing
            rec_model = None
            if hasattr(self.app, 'models') and 'recognition' in self.app.models:
                rec_model = self.app.models['recognition']
            
            for i in range(batch_size):
                img = x_np[i]
                # Denormalize: [-1, 1] -> [0, 1] -> [0, 255]
                img = (img + 1.0) / 2.0
                img = (img * 255.0).astype(np.uint8)
                
                # Convert CHW -> HWC
                img = img.transpose(1, 2, 0)  # CHW -> HWC
                
                # Convert RGB to BGR (InsightFace uses BGR)
                img = img[:, :, ::-1]  # RGB -> BGR
                
                # Get embedding
                if rec_model is not None:
                    # Direct recognition model call (faster, for pre-aligned images)
                    try:
                        # Prepare input for recognition model
                        # InsightFace models typically expect specific input format
                        embedding = rec_model.get(img)
                        if embedding is None:
                            embedding = np.zeros(self.embedding_size, dtype=np.float32)
                    except:
                        # Fallback to face detection
                        faces = self.app.get(img)
                        if len(faces) > 0:
                            embedding = faces[0].embedding
                        else:
                            embedding = np.zeros(self.embedding_size, dtype=np.float32)
                else:
                    # Use face detection (slower but more robust)
                    faces = self.app.get(img)
                    if len(faces) > 0:
                        embedding = faces[0].embedding
                    else:
                        embedding = np.zeros(self.embedding_size, dtype=np.float32)
                
                embeddings_list.append(embedding)
            
            # Stack embeddings
            embeddings = np.stack(embeddings_list, axis=0)
            embeddings = torch.from_numpy(embeddings).float()
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings
        
        def extract_features(self, x):
            """
            Extract face embeddings (alias for forward).
            
            Args:
                x: Input images [batch_size, 3, 112, 112]
            
            Returns:
                Face embeddings [batch_size, embedding_size]
            """
            return self.forward(x)
    
    wrapper = InsightFaceWrapper(app)
    return wrapper


def load_pretrained_iresnet(backbone_name='iresnet50', pretrained_path=None, num_features=512):
    """
    Load pretrained iResNet model.
    
    Args:
        backbone_name: Backbone name ('iresnet18', 'iresnet50', 'iresnet100')
        pretrained_path: Path to pretrained weights
        num_features: Embedding dimension
    
    Returns:
        FaceRecognitionModel instance
    """
    # Create backbone
    if backbone_name == 'iresnet18':
        backbone = iresnet18(num_features=num_features)
    elif backbone_name == 'iresnet50':
        backbone = iresnet50(num_features=num_features)
    elif backbone_name == 'iresnet100':
        backbone = iresnet100(num_features=num_features)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    # Create face recognition model
    model = FaceRecognitionModel(backbone, embedding_size=num_features)
    
    return model


def load_model_from_config(config):
    """
    Load model based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Model instance
    """
    model_config = config.get('model', {})
    use_insightface = model_config.get('use_insightface', False)
    model_name = model_config.get('insightface_model', 'buffalo_l')
    
    if use_insightface:
        # Load from InsightFace
        app = load_insightface_model(model_name)
        model = extract_model_from_insightface(app)
    else:
        # Load iResNet
        backbone_name = model_config.get('backbone', 'iresnet50')
        pretrained_path = model_config.get('pretrained_path', None)
        num_features = model_config.get('embedding_size', 512)
        model = load_pretrained_iresnet(backbone_name, pretrained_path, num_features)
    
    return model


def get_embedding_function(model, device='cuda'):
    """
    Get embedding function from model.
    
    Args:
        model: Face recognition model
        device: Device to run on
    
    Returns:
        Function that takes images and returns embeddings
    """
    model = model.to(device)
    model.eval()
    
    def embed(images):
        """
        Extract face embeddings.
        
        Args:
            images: Batch of images [batch_size, 3, 112, 112]
        
        Returns:
            Embeddings [batch_size, embedding_size]
        """
        with torch.no_grad():
            if isinstance(images, torch.Tensor):
                images = images.to(device)
            else:
                # Convert numpy to tensor
                images = torch.from_numpy(images).to(device)
            
            embeddings = model(images)
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    return embed
