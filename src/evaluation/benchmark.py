"""
Benchmark runner for all evaluation datasets
"""

import yaml
import torch
from pathlib import Path
from .verification import evaluate_lfw, evaluate_verification
from .metrics import compute_all_metrics
from ..data.dataloader import get_lfw_dataloader


class BenchmarkRunner:
    """
    Runner for comprehensive evaluation on all benchmarks.
    """
    
    def __init__(self, model, config_path=None, device='cuda'):
        """
        Initialize benchmark runner.
        
        Args:
            model: Face recognition model to evaluate
            config_path: Path to config file
            device: Device to run on
        """
        self.model = model
        self.device = device
        
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    
    def evaluate_lfw(self, root_dir=None, pairs_file=None):
        """Evaluate on LFW dataset."""
        root_dir = root_dir or self.config.get('data', {}).get('evaluation', {}).get('lfw')
        if not root_dir:
            raise ValueError("LFW root directory not specified")
        
        # Default pairs file location
        if pairs_file is None:
            pairs_file = Path(root_dir) / 'pairs.txt'
        
        return evaluate_lfw(self.model, root_dir, pairs_file, self.device)
    
    def evaluate_cfp_fp(self, root_dir=None, pairs_file=None):
        """Evaluate on CFP-FP dataset."""
        root_dir = root_dir or self.config.get('data', {}).get('evaluation', {}).get('cfp_fp')
        if not root_dir:
            raise ValueError("CFP-FP root directory not specified")
        
        # Similar to LFW evaluation
        from ..data.dataset import LFWDataset
        from torch.utils.data import DataLoader
        
        dataset = LFWDataset(root_dir, pairs_file)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        from .verification import evaluate_verification
        return evaluate_verification(self.model, dataloader, self.device)
    
    def evaluate_all(self, datasets=None):
        """
        Evaluate on all specified datasets.
        
        Args:
            datasets: List of dataset names to evaluate (None for all)
        
        Returns:
            Dictionary with results for each dataset
        """
        results = {}
        
        # Default datasets
        if datasets is None:
            datasets = ['lfw', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw']
        
        # Evaluate each dataset
        if 'lfw' in datasets:
            try:
                results['lfw'] = self.evaluate_lfw()
            except Exception as e:
                results['lfw'] = {'error': str(e)}
        
        if 'cfp_fp' in datasets:
            try:
                results['cfp_fp'] = self.evaluate_cfp_fp()
            except Exception as e:
                results['cfp_fp'] = {'error': str(e)}
        
        # Add other datasets similarly
        # IJB-B and IJB-C require more complex evaluation
        
        return results
    
    def compare_models(self, original_model, unlearned_model, datasets=None):
        """
        Compare original and unlearned models on all benchmarks.
        
        Args:
            original_model: Original model
            unlearned_model: Unlearned model
            datasets: List of datasets to evaluate
        
        Returns:
            Dictionary with comparison results
        """
        # Evaluate original model
        self.model = original_model
        original_results = self.evaluate_all(datasets)
        
        # Evaluate unlearned model
        self.model = unlearned_model
        unlearned_results = self.evaluate_all(datasets)
        
        # Compute differences
        comparison = {}
        for dataset in original_results.keys():
            if 'error' in original_results[dataset] or 'error' in unlearned_results[dataset]:
                continue
            
            comparison[dataset] = {}
            for metric in original_results[dataset].keys():
                if metric in unlearned_results[dataset]:
                    orig_val = original_results[dataset][metric]
                    unlearn_val = unlearned_results[dataset][metric]
                    
                    if isinstance(orig_val, (int, float)) and isinstance(unlearn_val, (int, float)):
                        comparison[dataset][metric] = {
                            'original': orig_val,
                            'unlearned': unlearn_val,
                            'difference': unlearn_val - orig_val,
                            'relative_change': (unlearn_val - orig_val) / orig_val if orig_val != 0 else 0
                        }
        
        return {
            'original': original_results,
            'unlearned': unlearned_results,
            'comparison': comparison
        }
