#!/usr/bin/env python
"""
Script to run unlearning experiments.
"""

import argparse
import yaml
import torch
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from utils.model_loader import load_model_from_config
from linearizer.linearizer import Linearizer
from unlearning.unlearning import UnlearningEngine
from data.dataloader import get_ms1mv2_dataloader


def main():
    parser = argparse.ArgumentParser(description='Run Unlearning Experiments')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--linearizer-checkpoint', type=str, required=True,
                       help='Path to trained linearizer checkpoint')
    parser.add_argument('--identity-ids', type=int, nargs='+', default=None,
                       help='List of identity IDs to unlearn')
    parser.add_argument('--method', type=str, default=None,
                       choices=['orthogonal_projection', 'svd_update', 'linear_combination'],
                       help='Unlearning method')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    
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
    
    # Load Linearizer
    print(f"Loading Linearizer from {args.linearizer_checkpoint}...")
    linearizer_config = config['linearizer']
    linearizer = Linearizer(
        model=model,
        embedding_size=embedding_size,
        num_blocks=linearizer_config.get('num_blocks', 4),
        hidden_dim=linearizer_config.get('hidden_dim', 1024)
    )
    linearizer = linearizer.to(device)
    
    checkpoint = torch.load(args.linearizer_checkpoint, map_location=device)
    linearizer.load_state_dict(checkpoint['state_dict'])
    linearizer.eval()
    
    # Get identity IDs to unlearn
    identity_ids_to_unlearn = args.identity_ids
    if identity_ids_to_unlearn is None:
        identity_ids_to_unlearn = config['unlearning'].get('target_identities', [])
        if len(identity_ids_to_unlearn) == 0:
            print("Error: No identity IDs specified. Use --identity-ids or set in config.yaml")
            return
    
    print(f"Identities to unlearn: {identity_ids_to_unlearn}")
    
    # Get unlearning method
    method = args.method or config['unlearning'].get('method', 'orthogonal_projection')
    print(f"Unlearning method: {method}")
    
    # Create unlearning engine
    unlearning_engine = UnlearningEngine(linearizer, method=method)
    
    # Load data
    print("Loading data...")
    ms1mv2_path = config['data']['ms1mv2']['path']
    dataloader = get_ms1mv2_dataloader(
        ms1mv2_path,
        batch_size=64,
        is_training=True
    )
    
    # Perform unlearning
    print("Performing unlearning...")
    updated_operator = unlearning_engine.unlearn(
        dataloader,
        identity_ids_to_unlearn,
        device=device
    )
    
    print("Unlearning completed!")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        'identity_ids_unlearned': identity_ids_to_unlearn,
        'method': method,
        'linear_operator_shape': list(updated_operator.shape) if updated_operator is not None else None
    }
    
    results_path = os.path.join(args.output_dir, 'unlearning_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Save unlearned linearizer
    checkpoint_path = os.path.join(args.output_dir, 'unlearned_linearizer.pth')
    torch.save({
        'state_dict': linearizer.state_dict(),
        'config': config,
        'unlearned_identities': identity_ids_to_unlearn
    }, checkpoint_path)
    print(f"Unlearned linearizer saved to {checkpoint_path}")


if __name__ == '__main__':
    main()
