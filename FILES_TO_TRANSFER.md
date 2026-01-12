# Files to Transfer to Server

## Essential Files (Must Transfer)

### Source Code
```
src/
├── __init__.py
├── linearizer/
│   ├── __init__.py
│   ├── invertible_net.py      # ImageToLatentNetwork, LatentToEmbeddingNetwork
│   ├── linearizer.py           # Main Linearizer class
│   ├── linearization.py        # LinearizedModel
│   └── utils.py
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   ├── face_recognition.py
│   └── losses.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataloader.py
│   └── preprocessing.py
├── utils/
│   ├── __init__.py
│   └── model_loader.py         # InsightFace integration
├── evaluation/
│   └── ... (all files)
└── unlearning/
    └── ... (all files)
```

### Configuration & Scripts
```
config.yaml                    # Main configuration file (UPDATE PATHS!)
requirements.txt               # Python dependencies
scripts/
└── train_linearizer.py        # Training script
```

### Notebooks
```
notebooks/
├── 00_quick_setup.ipynb       # Setup verification (RUN THIS FIRST!)
└── 03_linearization.ipynb     # Main training notebook
```

### Documentation
```
README.md                      # Project overview
SETUP_GUIDE.md                 # Detailed setup instructions
QUICK_START_SERVER.md          # Quick reference
IMPLEMENTATION_SUMMARY.md      # Implementation details
```

### Helper Scripts
```
quick_start.py                 # Setup verification script
transfer_to_server.sh          # Transfer helper (optional)
```

## Files NOT to Transfer (Will be created on server)

```
checkpoints/                   # Model checkpoints (created during training)
results/                       # Evaluation results (created during evaluation)
data/                          # Dataset (already on server)
.git/                          # Git metadata (optional)
__pycache__/                   # Python cache (auto-generated)
*.pyc                          # Compiled Python files (auto-generated)
.ipynb_checkpoints/            # Jupyter checkpoints (auto-generated)
```

## Transfer Methods

### Method 1: SCP (Secure Copy)
```bash
# From your local machine:
scp -r src/ notebooks/ scripts/ *.yaml *.txt *.py *.md \
    username@server:/path/to/destination/fast-unlearning-face-recognition/
```

### Method 2: Git (Recommended)
```bash
# On local machine:
git add .
git commit -m "Linearizer implementation"
git push

# On server:
git clone <your-repo-url>
cd fast-unlearning-face-recognition
```

### Method 3: Jupyter Upload
1. Create zip file: `zip -r project.zip src/ notebooks/ scripts/ *.yaml *.txt *.py *.md`
2. Upload via Jupyter interface
3. Extract: `!unzip project.zip`

### Method 4: Using transfer_to_server.sh
```bash
# Make executable:
chmod +x transfer_to_server.sh

# Run:
./transfer_to_server.sh username@server:/destination/path
```

## After Transfer Checklist

- [ ] All `src/` subdirectories transferred
- [ ] `config.yaml` present
- [ ] `requirements.txt` present
- [ ] `notebooks/00_quick_setup.ipynb` present
- [ ] `notebooks/03_linearization.ipynb` present
- [ ] `scripts/train_linearizer.py` present
- [ ] Update `config.yaml` with server dataset path
- [ ] Run `python quick_start.py` to verify

## Directory Structure on Server

After transfer, your server should have:
```
fast-unlearning-face-recognition/
├── src/                       ✓
│   ├── linearizer/            ✓
│   ├── models/                ✓
│   ├── data/                  ✓
│   ├── utils/                 ✓
│   ├── evaluation/            ✓
│   └── unlearning/           ✓
├── notebooks/                 ✓
│   ├── 00_quick_setup.ipynb   ✓
│   └── 03_linearization.ipynb ✓
├── scripts/                   ✓
│   └── train_linearizer.py     ✓
├── config.yaml                ✓ (UPDATE PATHS!)
├── requirements.txt           ✓
├── quick_start.py             ✓
└── *.md                       ✓ (documentation)
```

## Quick Verification

After transfer, run this on server:
```bash
cd fast-unlearning-face-recognition
python quick_start.py
```

This will verify:
- ✓ All files present
- ✓ Dependencies installed
- ✓ InsightFace model available
- ✓ Dataset path correct
- ✓ Model loading works
- ✓ Linearizer creation works
