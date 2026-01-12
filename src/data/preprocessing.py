"""
Image preprocessing utilities for face recognition
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def get_face_alignment_transform(image_size=(112, 112)):
    """
    Get face alignment transform following InsightFace standards.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Transform function
    """
    def align_face(img, landmarks=None):
        """
        Align face using landmarks or center crop.
        
        Args:
            img: Input image (numpy array or PIL Image)
            landmarks: Optional face landmarks for alignment
        
        Returns:
            Aligned face image
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        if img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # If landmarks provided, use them for alignment
        if landmarks is not None:
            # Use similarity transform for alignment
            # This is a simplified version - full implementation would use 5-point landmarks
            aligned = cv2.resize(img, image_size)
        else:
            # Center crop and resize
            h, w = img.shape[:2]
            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            cropped = img[start_h:start_h+size, start_w:start_w+size]
            aligned = cv2.resize(cropped, image_size)
        
        return aligned
    
    return align_face


def get_train_transform(image_size=(112, 112)):
    """
    Get training data augmentation transform.
    
    Args:
        image_size: Target image size
    
    Returns:
        torchvision transform
    """
    return transforms.Compose([
        transforms.Resize((image_size[0] + 10, image_size[1] + 10)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transform(image_size=(112, 112)):
    """
    Get validation/test data transform (no augmentation).
    
    Args:
        image_size: Target image size
    
    Returns:
        torchvision transform
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def normalize_image(img):
    """
    Normalize image to [-1, 1] range.
    
    Args:
        img: Input image tensor or numpy array
    
    Returns:
        Normalized image
    """
    if isinstance(img, np.ndarray):
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
    elif isinstance(img, torch.Tensor):
        img = img.float() / 255.0
        img = (img - 0.5) / 0.5
    
    return img


def denormalize_image(img):
    """
    Denormalize image from [-1, 1] to [0, 255].
    
    Args:
        img: Normalized image tensor or numpy array
    
    Returns:
        Denormalized image
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    img = (img * 0.5) + 0.5
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    return img
