#!/usr/bin/env python
"""
Script to evaluate models on face recognition benchmarks.
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
from evaluation.benchmark import BenchmarkRunner
from evaluation.verification import evaluate_lfw


def main():
    parser = argparse.ArgumentParser(description='Evaluate Models')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--model-checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--linearizer-checkpoint', type=str, default=None,
                       help='Path to linearizer checkpoint (optional)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['lfw'],
                       help='Datasets to evaluate on')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() and config['device']['cuda'] else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    if args.linearizer_checkpoint:
        print("Loading Linearizer model...")
        model = load_model_from_config(config)
        model = model.to(device)
        model.eval()
        
        embedding_size = config['model'].get('embedding_size', 512)
        linearizer_config = config['linearizer']
        linearizer = Linearizer(
            model=model,
            embedding_size=embedding_size,
            num_blocks=linearizer_config.get('num_blocks', 4)
        )
        linearizer = linearizer.to(device)
        
        checkpoint = torch.load(args.linearizer_checkpoint, map_location=device)
        linearizer.load_state_dict(checkpoint['state_dict'])
        linearizer.eval()
        
        eval_model = linearizer
        model_type = 'linearized'
    else:
        print("Loading original model...")
        eval_model = load_model_from_config(config)
        eval_model = eval_model.to(device)
        eval_model.eval()
        model_type = 'original'
    
    # Evaluate on datasets
    results = {}
    
    for dataset_name in args.datasets:
        print(f"\nEvaluating on {dataset_name}...")
        
        try:
            if dataset_name == 'lfw':
                lfw_path = config['data']['evaluation']['lfw']
                lfw_pairs_file = os.path.join(lfw_path, 'pairs.txt')
                
                if os.path.exists(lfw_path):
                    dataset_results = evaluate_lfw(
                        eval_model, lfw_path, lfw_pairs_file, device
                    )
                    results[dataset_name] = dataset_results
                    print(f"{dataset_name} results:")
                    for key, value in dataset_results.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for k, v in value.items():
                                print(f"    {k}: {v:.4f}")
                        else:
                            print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {dataset_name} dataset not found at {lfw_path}")
                    results[dataset_name] = {'error': 'Dataset not found'}
            else:
                print(f"  Dataset {dataset_name} not yet implemented")
                results[dataset_name] = {'error': 'Not implemented'}
        
        except Exception as e:
            print(f"  Error evaluating {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f'evaluation_{model_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
