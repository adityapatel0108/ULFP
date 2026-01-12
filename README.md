# Fast Unlearning on Linearized Face Recognition Models

This project implements fast machine unlearning for face recognition models by adapting the Linearizer framework. The approach linearizes trained neural networks into a learned latent space, enabling efficient unlearning through simple linear operations instead of expensive retraining.

## Project Overview

The core innovation is using invertible neural networks to create a learned latent space where face recognition models become linear operators. This enables direct linear algebra operations (SVD, orthogonal projections) for efficient unlearning of specific identities.

## Features

- **Linearizer Framework**: Transform face recognition models into linear operators in learned latent space
- **Fast Unlearning**: Remove specific identity influences using linear operations
- **Comprehensive Evaluation**: Standard face recognition metrics and unlearning-specific metrics
- **Jupyter Notebooks**: Interactive workflow for experimentation

## Project Structure

```
fast-unlearning-face-recognition/
├── src/
│   ├── data/           # Dataset loaders and preprocessing
│   ├── models/         # Face recognition models (iResNet, ArcFace)
│   ├── linearizer/     # Linearizer framework implementation
│   ├── unlearning/     # Unlearning operations
│   ├── evaluation/     # Evaluation metrics and benchmarks
│   └── utils/          # Utility functions
├── notebooks/          # Jupyter notebooks for interactive workflow
├── scripts/            # Execution scripts
├── data/               # Dataset storage
├── checkpoints/        # Model checkpoints
└── results/            # Evaluation results
```

## Installation

### Local Setup

1. Clone the repository (or navigate to the project directory)

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
   - MS1MV2 training dataset from [InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
   - Evaluation datasets (LFW, CFP-FP, AgeDB-30, CALFW, CPLFW, IJB-B, IJB-C)

4. Download pretrained models from [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)

### Server Setup (College Jupyter Server)

**Quick Start**: See [`QUICK_START_SERVER.md`](QUICK_START_SERVER.md) for a 5-minute setup guide.

**Detailed Guide**: See [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for comprehensive instructions.

**Steps**:
1. Transfer files to server (SCP, Git, or Jupyter upload)
2. Update `config.yaml` with your dataset path
3. Install dependencies: `pip install -r requirements.txt`
4. Verify setup: `python quick_start.py` or run `notebooks/00_quick_setup.ipynb`
5. Start training: `notebooks/03_linearization.ipynb` or `scripts/train_linearizer.py`

## Usage

### Jupyter Notebooks

Start with the notebooks in order:
1. `notebooks/01_data_exploration.ipynb` - Explore datasets
2. `notebooks/02_model_loading.ipynb` - Load pretrained models
3. `notebooks/03_linearization.ipynb` - Implement and test linearization
4. `notebooks/04_unlearning.ipynb` - Run unlearning experiments
5. `notebooks/05_evaluation.ipynb` - Comprehensive evaluation
6. `notebooks/06_comparison.ipynb` - Compare with retraining

### Scripts

```bash
# Train linearizer on face recognition model
python scripts/train_linearizer.py --config config.yaml

# Run unlearning experiments
python scripts/run_unlearning.py --config config.yaml

# Evaluate models
python scripts/evaluate.py --config config.yaml

# Compare unlearning vs retraining
python scripts/compare_methods.py --config config.yaml
```

## Datasets

- **Training**: MS1MV2 (5M images, 85K identities)
- **Evaluation**: LFW, CFP-FP, AgeDB-30, CALFW, CPLFW, IJB-B, IJB-C

## Models

- **Backbones**: iResNet18, iResNet50, iResNet100
- **Losses**: ArcFace, MagFace, AdaFace
- Pretrained models available from InsightFace Model Zoo

## Evaluation Metrics

- **Face Recognition**: Verification accuracy, FNMR@FMR, EER, AUC
- **Unlearning**: Retain accuracy, Forget accuracy, Membership Inference Attack, Speed

## References

- Berman, Nimrod, Assaf Hallak, and Assaf Shocher. "Who Said Neural Networks Aren't Linear?." arXiv preprint arXiv:2510.08570 (2025).
- InsightFace: https://github.com/deepinsight/insightface

## License

This project is for research purposes.
