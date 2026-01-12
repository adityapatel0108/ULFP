#!/usr/bin/env python
"""
Quick start script to verify setup and run Linearizer training.
Run this on your college Jupyter server to check everything is ready.
"""

import os
import sys
import yaml
import torch

def check_dependencies():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("Checking Dependencies...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'insightface': 'InsightFace',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def check_insightface_model():
    """Check if InsightFace buffalo_l model is available."""
    print("\n" + "=" * 60)
    print("Checking InsightFace Model...")
    print("=" * 60)
    
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✓ InsightFace buffalo_l model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to load InsightFace model: {e}")
        print("  Try: insightface.model_zoo.download('buffalo_l')")
        return False

def check_dataset_path(config):
    """Check if dataset path exists."""
    print("\n" + "=" * 60)
    print("Checking Dataset Path...")
    print("=" * 60)
    
    dataset_path = config.get('data', {}).get('ms1mv2', {}).get('path', '')
    
    if not dataset_path:
        print("✗ Dataset path not specified in config.yaml")
        return False
    
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found at: {dataset_path}")
        
        # Check for expected structure
        images_dir = os.path.join(dataset_path, 'images')
        if os.path.exists(images_dir):
            print(f"✓ Images directory found")
        else:
            print(f"⚠ Images directory not found (expected: {images_dir})")
        
        return True
    else:
        print(f"✗ Dataset NOT found at: {dataset_path}")
        print(f"  Please update config.yaml with correct path")
        return False

def check_gpu():
    """Check GPU availability."""
    print("\n" + "=" * 60)
    print("Checking GPU...")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ CUDA not available, will use CPU (slower)")
        return False

def check_project_structure():
    """Check if project files are in place."""
    print("\n" + "=" * 60)
    print("Checking Project Structure...")
    print("=" * 60)
    
    required_files = [
        'src/linearizer/linearizer.py',
        'src/linearizer/invertible_net.py',
        'src/utils/model_loader.py',
        'src/data/dataset.py',
        'config.yaml',
        'scripts/train_linearizer.py',
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} NOT found")
            all_present = False
    
    return all_present

def load_config():
    """Load and validate configuration."""
    print("\n" + "=" * 60)
    print("Loading Configuration...")
    print("=" * 60)
    
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"✗ config.yaml not found at: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return None

def main():
    """Main setup verification."""
    print("\n" + "=" * 60)
    print("Linearizer Framework - Setup Verification")
    print("=" * 60)
    print("\nThis script will verify your setup is ready to run.")
    print("Please fix any issues before proceeding.\n")
    
    # Check project structure
    if not check_project_structure():
        print("\n✗ Project structure incomplete. Please check file paths.")
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Please install missing dependencies first.")
        return False
    
    # Load config
    config = load_config()
    if config is None:
        return False
    
    # Check GPU
    check_gpu()
    
    # Check InsightFace
    if not check_insightface_model():
        print("\n⚠ InsightFace model issue. Training may fail.")
    
    # Check dataset
    if not check_dataset_path(config):
        print("\n✗ Dataset path issue. Please update config.yaml")
        return False
    
    # Final summary
    print("\n" + "=" * 60)
    print("Setup Verification Complete!")
    print("=" * 60)
    print("\n✓ All checks passed! You're ready to run training.")
    print("\nNext steps:")
    print("1. Open notebooks/03_linearization.ipynb in Jupyter")
    print("2. Or run: python scripts/train_linearizer.py --config config.yaml")
    print("\n" + "=" * 60)
    
    return True

if __name__ == '__main__':
    # Add src to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    success = main()
    sys.exit(0 if success else 1)
