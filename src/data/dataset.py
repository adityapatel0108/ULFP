"""
Dataset loading classes for face recognition
"""

import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import get_train_transform, get_val_transform


class MS1MV2Dataset(Dataset):
    """
    MS1MV2 face recognition dataset loader.
    
    Expected structure:
    data/
    ├── ms1m-retinaface-t1/
    │   ├── images/
    │   │   ├── id1/
    │   │   │   ├── img1.jpg
    │   │   │   └── img2.jpg
    │   │   └── id2/
    │   └── meta/
    │       └── identity_list.txt
    """
    
    def __init__(self, root_dir, transform=None, is_training=True):
        """
        Initialize MS1MV2 dataset.
        
        Args:
            root_dir: Root directory of MS1MV2 dataset
            transform: Optional transform to apply
            is_training: Whether this is training data (affects augmentation)
        """
        self.root_dir = root_dir
        self.is_training = is_training
        
        if transform is None:
            if is_training:
                transform = get_train_transform()
            else:
                transform = get_val_transform()
        self.transform = transform
        
        # Load identity list
        self.identities = []
        self.samples = []  # List of (image_path, identity_id)
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset structure."""
        images_dir = os.path.join(self.root_dir, 'images')
        meta_dir = os.path.join(self.root_dir, 'meta')
        
        # Try to load from cached list if available
        cache_file = os.path.join(self.root_dir, 'dataset_cache.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.identities = cache['identities']
                self.samples = cache['samples']
            return
        
        # Load identity list
        identity_file = os.path.join(meta_dir, 'identity_list.txt')
        if os.path.exists(identity_file):
            with open(identity_file, 'r') as f:
                self.identities = [line.strip() for line in f.readlines()]
        else:
            # If no identity list, scan directory
            if os.path.exists(images_dir):
                self.identities = sorted([d for d in os.listdir(images_dir) 
                                        if os.path.isdir(os.path.join(images_dir, d))])
        
        # Create identity to ID mapping
        identity_to_id = {identity: idx for idx, identity in enumerate(self.identities)}
        
        # Load all samples
        if os.path.exists(images_dir):
            for identity in self.identities:
                identity_dir = os.path.join(images_dir, identity)
                if not os.path.isdir(identity_dir):
                    continue
                
                identity_id = identity_to_id[identity]
                for img_name in os.listdir(identity_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(identity_dir, img_name)
                        self.samples.append((img_path, identity_id))
        
        # Cache the dataset structure
        if not os.path.exists(cache_file):
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'identities': self.identities,
                    'samples': self.samples
                }, f)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, identity_id = self.samples[idx]
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image is corrupted, return a black image
            img = Image.new('RGB', (112, 112), color=(0, 0, 0))
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        return img, identity_id
    
    def get_identity_samples(self, identity_id):
        """Get all samples for a specific identity."""
        return [(path, id) for path, id in self.samples if id == identity_id]


class FaceVerificationDataset(Dataset):
    """
    Base class for face verification datasets (LFW, CFP-FP, etc.).
    """
    
    def __init__(self, root_dir, pairs_file=None, transform=None):
        """
        Initialize verification dataset.
        
        Args:
            root_dir: Root directory of dataset
            pairs_file: Path to pairs file (format depends on dataset)
            transform: Optional transform to apply
        """
        self.root_dir = root_dir
        self.transform = transform or get_val_transform()
        self.pairs = []
        
        if pairs_file:
            self._load_pairs(pairs_file)
    
    def _load_pairs(self, pairs_file):
        """Load pairs file. To be implemented by subclasses."""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img1_path, img2_path, is_same = pair
        
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, is_same


class LFWDataset(FaceVerificationDataset):
    """LFW (Labeled Faces in the Wild) dataset."""
    
    def _load_pairs(self, pairs_file):
        """Load LFW pairs file."""
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header if present
        start_idx = 0
        if not lines[0][0].isdigit():
            start_idx = 1
        
        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) == 3:
                # Same person: name image1 image2
                name, img1, img2 = parts
                img1_path = os.path.join(self.root_dir, name, f"{name}_{img1.zfill(4)}.jpg")
                img2_path = os.path.join(self.root_dir, name, f"{name}_{img2.zfill(4)}.jpg")
                self.pairs.append((img1_path, img2_path, True))
            elif len(parts) == 4:
                # Different people: name1 image1 name2 image2
                name1, img1, name2, img2 = parts
                img1_path = os.path.join(self.root_dir, name1, f"{name1}_{img1.zfill(4)}.jpg")
                img2_path = os.path.join(self.root_dir, name2, f"{name2}_{img2.zfill(4)}.jpg")
                self.pairs.append((img1_path, img2_path, False))


class IJBDataset(Dataset):
    """
    IJB-B and IJB-C datasets loader.
    These datasets have a more complex structure with templates.
    """
    
    def __init__(self, root_dir, protocol_file, transform=None):
        """
        Initialize IJB dataset.
        
        Args:
            root_dir: Root directory of IJB dataset
            protocol_file: Path to protocol file
            transform: Optional transform to apply
        """
        self.root_dir = root_dir
        self.transform = transform or get_val_transform()
        self.templates = []
        
        self._load_protocol(protocol_file)
    
    def _load_protocol(self, protocol_file):
        """Load IJB protocol file. Simplified version."""
        # IJB datasets have complex protocol files
        # This is a placeholder - full implementation would parse the actual format
        pass
    
    def __len__(self):
        return len(self.templates)
    
    def __getitem__(self, idx):
        # Placeholder implementation
        template = self.templates[idx]
        # Load template images and metadata
        return template
