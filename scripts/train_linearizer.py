#!/usr/bin/env python
"""
Script to train the Linearizer framework on a face recognition model.
"""

import argparse
import yaml
import torch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from utils.model_loader import load_model_from_config
from linearizer.linearizer import Linearizer
from data.dataloader import get_ms1mv2_dataloader


def main():
    parser = argparse.ArgumentParser(description='Train Linearizer')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() and config['device']['cuda'] else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading face recognition model...")
    model = load_model_from_config(config)
    model = model.to(device)
    model.eval()
    
    embedding_size = config['model'].get('embedding_size', 512)
    
    # Create Linearizer with sandwich architecture
    print("Creating Linearizer with sandwich architecture...")
    linearizer_config = config['linearizer']
    latent_dim = linearizer_config.get('latent_dim', 512)
    image_size = (112, 112)  # Standard face recognition image size
    
    linearizer = Linearizer(
        model=model,
        embedding_size=embedding_size,
        latent_dim=latent_dim,
        num_blocks=linearizer_config.get('num_blocks', 4),
        hidden_dim=linearizer_config.get('hidden_dim', 1024),
        num_layers=linearizer_config.get('num_layers', 3),
        image_size=image_size
    )
    linearizer = linearizer.to(device)
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        linearizer.load_state_dict(checkpoint['state_dict'])
    
    # Load training data
    print("Loading training data...")
    ms1mv2_path = config['data']['ms1mv2']['path']
    dataloader = get_ms1mv2_dataloader(
        ms1mv2_path,
        batch_size=linearizer_config.get('batch_size', 64),
        num_workers=config['training'].get('num_workers', 4),
        is_training=True
    )
    
    # Train
    print("Training Linearizer...")
    checkpoint_dir = args.checkpoint_dir or linearizer_config.get('checkpoint_dir', './checkpoints/linearizer')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    linearizer.train_linearizer(
        dataloader,
        num_epochs=linearizer_config.get('num_epochs', 100),
        lr=linearizer_config.get('learning_rate', 0.0001),
        device=device
    )
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'linearizer_final.pth')
    torch.save({
        'state_dict': linearizer.state_dict(),
        'config': config,
        'embedding_size': embedding_size
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()
