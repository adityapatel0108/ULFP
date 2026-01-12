# Linearizer Implementation Summary

## Overview

This implementation creates a **Linearizer framework** for face recognition models using the **sandwich architecture** as described in:

**Berman, Nimrod, Assaf Hallak, and Assaf Shocher. "Who Said Neural Networks Aren't Linear?." arXiv preprint arXiv:2510.08570 (2025).**

## Sandwich Architecture

The core innovation is the sandwich architecture:

```
f(x) = g⁻¹ᵧ(Agₓ(x))
```

Where:
- **gₓ**: Image → Latent (invertible network mapping images to latent space)
- **A**: Linear operator in latent space (learnable matrix)
- **g⁻¹ᵧ**: Latent → Embedding (invertible network mapping latent space to embeddings)

This allows us to perform linear operations (SVD, projections, etc.) on the face recognition model in the learned latent space.

## Implementation Details

### 1. InsightFace Integration (`src/utils/model_loader.py`)

- **`load_insightface_model()`**: Loads InsightFace buffalo_l model
- **`extract_model_from_insightface()`**: Creates a PyTorch wrapper around InsightFace model
  - Handles image format conversion (RGB→BGR, normalization)
  - Extracts embeddings from pre-aligned face images
  - Supports batch processing

### 2. Invertible Networks (`src/linearizer/invertible_net.py`)

#### `ImageToLatentNetwork` (gₓ)
- Maps images `[B, 3, 112, 112]` → latent vectors `[B, latent_dim]`
- Uses CNN encoder + invertible blocks
- Architecture:
  - CNN encoder: 7×7 conv → 3×3 convs → GlobalAvgPool → Linear
  - Invertible blocks: RealNVP-style coupling layers

#### `LatentToEmbeddingNetwork` (g⁻¹ᵧ)
- Maps latent vectors `[B, latent_dim]` → embeddings `[B, embedding_dim]`
- Uses invertible blocks (RealNVP coupling layers)
- Handles dimension mismatches with projection layers

#### `InvertibleBlock`
- RealNVP-style affine coupling layers
- Ensures invertibility through exponential scaling
- Uses tanh activation for scale factors

### 3. Linearizer Class (`src/linearizer/linearizer.py`)

Main class implementing the sandwich architecture:

```python
class Linearizer(nn.Module):
    def __init__(self, model, embedding_size, latent_dim, ...):
        self.g_x = ImageToLatentNetwork(...)      # Image → Latent
        self.linear_op = nn.Linear(...)           # Linear operator A
        self.g_y_inv = LatentToEmbeddingNetwork(...)  # Latent → Embedding
    
    def forward(self, x):
        z = self.g_x(x)                    # gₓ(x)
        z = self.linear_op(z)              # A(z)
        return self.g_y_inv(z)            # g⁻¹ᵧ(A(z))
```

### 4. Training (`scripts/train_linearizer.py`)

Training process:
1. Load InsightFace buffalo_l model (frozen, for reference embeddings)
2. Create Linearizer with sandwich architecture
3. Train on MS1MV2 dataset:
   - Forward pass: `linearized_emb = linearizer(images)`
   - Loss: `MSE(linearized_emb, original_emb)`
   - Optimize: gₓ, A, and g⁻¹ᵧ parameters

## Usage

### Configuration (`config.yaml`)

```yaml
model:
  use_insightface: true
  insightface_model: "buffalo_l"
  embedding_size: 512

linearizer:
  latent_dim: 512
  num_blocks: 4
  hidden_dim: 1024
  num_layers: 3
  learning_rate: 0.0001
  batch_size: 64
  num_epochs: 100
```

### Training

```bash
python scripts/train_linearizer.py --config config.yaml
```

### Jupyter Notebook

See `notebooks/03_linearization.ipynb` for interactive workflow:
1. Load InsightFace model
2. Create Linearizer
3. Train on MS1MV2 dataset
4. Visualize sandwich architecture flow
5. Evaluate reconstruction quality

## Key Features

1. **Sandwich Architecture**: Proper implementation of f(x) = g⁻¹ᵧ(Agₓ(x))
2. **InsightFace Integration**: Works with buffalo_l model from InsightFace
3. **Invertible Networks**: RealNVP-style coupling layers ensure invertibility
4. **Linear Operations**: Enables SVD, projections, and other linear algebra operations in latent space
5. **MS1MV2 Support**: Works with MS1MV2 dataset format

## File Structure

```
src/
├── linearizer/
│   ├── invertible_net.py      # ImageToLatentNetwork, LatentToEmbeddingNetwork
│   ├── linearizer.py           # Main Linearizer class
│   └── linearization.py        # LinearizedModel (legacy, kept for compatibility)
├── utils/
│   └── model_loader.py         # InsightFace model loading
└── data/
    └── dataset.py              # MS1MV2 dataset loader

scripts/
└── train_linearizer.py         # Training script

notebooks/
└── 03_linearization.ipynb     # Interactive workflow
```

## Next Steps

1. **Unlearning**: Use linear operations in latent space for fast unlearning
2. **Evaluation**: Test on face verification benchmarks (LFW, CFP-FP, etc.)
3. **Optimization**: Fine-tune hyperparameters for better reconstruction

## References

- Paper: [Who Said Neural Networks Aren't Linear?](https://arxiv.org/abs/2510.08570)
- GitHub: [Linearizer Framework](https://github.com/assafshocher/Linearizer)
- InsightFace: [Face Recognition Library](https://github.com/deepinsight/insightface)
