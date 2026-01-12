# Quick Start Guide for Server

## 🚀 Fast Setup (5 minutes)

### Step 1: Transfer Files
```bash
# Option A: Using SCP
scp -r fast-unlearning-face-recognition username@server:/path/to/destination/

# Option B: Using Git (if you have a repo)
git clone <your-repo-url>
cd fast-unlearning-face-recognition

# Option C: Upload zip via Jupyter interface
# Then: unzip fast-unlearning-face-recognition.zip
```

### Step 2: Update Config
Edit `config.yaml`:
```yaml
data:
  ms1mv2:
    path: "/your/server/path/to/ms1m-retinaface-t1"  # ← UPDATE THIS!
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup
```bash
# Option A: Run verification script
python quick_start.py

# Option B: Open Jupyter notebook
jupyter notebook notebooks/00_quick_setup.ipynb
```

### Step 5: Start Training
```bash
# Option A: Jupyter Notebook (Recommended)
jupyter notebook notebooks/03_linearization.ipynb

# Option B: Command line
python scripts/train_linearizer.py --config config.yaml
```

## 📋 Checklist

Before training, verify:
- [ ] Files transferred to server
- [ ] `config.yaml` updated with dataset path
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] InsightFace buffalo_l model available
- [ ] Dataset path exists and accessible
- [ ] GPU available (optional but recommended)

## 🔧 Common Issues

**"ModuleNotFoundError"**
```python
# Add to notebook cell:
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
```

**"Dataset not found"**
- Check path in `config.yaml`
- Use absolute path: `/full/path/to/dataset`

**"CUDA out of memory"**
- Reduce batch size in `config.yaml`:
  ```yaml
  linearizer:
    batch_size: 32  # Reduce from 64
  ```

## 📁 File Structure on Server

```
fast-unlearning-face-recognition/
├── src/                    # Source code
├── notebooks/              # Jupyter notebooks
│   ├── 00_quick_setup.ipynb    # Run this first!
│   └── 03_linearization.ipynb # Main training notebook
├── scripts/                # Training scripts
├── config.yaml             # Configuration (UPDATE THIS!)
├── requirements.txt        # Dependencies
└── quick_start.py          # Verification script
```

## 🎯 Training Workflow

1. **Setup** → Run `notebooks/00_quick_setup.ipynb`
2. **Train** → Run `notebooks/03_linearization.ipynb`
3. **Evaluate** → Check reconstruction quality
4. **Unlearn** → Use trained linearizer for unlearning

## 📞 Need Help?

1. Check `SETUP_GUIDE.md` for detailed instructions
2. Run `python quick_start.py` to diagnose issues
3. Check error messages - they usually tell you what's wrong

## ⏱️ Expected Training Time

- **Setup**: 5-10 minutes
- **Training**: 2-4 hours (depends on GPU and dataset size)
- **Evaluation**: 10-30 minutes

Good luck! 🎉
