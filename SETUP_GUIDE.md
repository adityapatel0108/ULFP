# Setup Guide for College Jupyter Server

This guide will help you transfer and run the Linearizer framework on your college Jupyter server.

## Prerequisites

Your server should have:
- ✅ MS1MV2 dataset (ms1m_arcface)
- ✅ InsightFace library installed
- ✅ buffalo_l model available
- ✅ Python 3.8+ with PyTorch
- ✅ Jupyter Notebook/Lab

## Step 1: Transfer Files to Server

### Option A: Using Git (Recommended)
```bash
# On your local machine, if you have a git repo:
git add .
git commit -m "Linearizer implementation"
git push

# On the server:
git clone <your-repo-url>
cd fast-unlearning-face-recognition
```

### Option B: Using SCP/SFTP
```bash
# From your local machine:
scp -r /Users/aditya/fast-unlearning-face-recognition username@server:/path/to/destination/

# Or use FileZilla/WinSCP for GUI transfer
```

### Option C: Using Jupyter Upload
1. Create a zip file of the project on your local machine
2. Upload via Jupyter's upload interface
3. Extract: `!unzip fast-unlearning-face-recognition.zip`

## Step 2: Verify Server Structure

After transferring, your server directory should look like:
```
fast-unlearning-face-recognition/
├── src/
│   ├── linearizer/
│   ├── models/
│   ├── data/
│   ├── utils/
│   └── ...
├── notebooks/
│   ├── 03_linearization.ipynb
│   └── ...
├── scripts/
│   └── train_linearizer.py
├── config.yaml
├── requirements.txt
└── ...
```

## Step 3: Update Configuration

Edit `config.yaml` to match your server paths:

```yaml
# Update dataset path to your server location
data:
  root_dir: "/path/to/your/data"  # Update this!
  ms1mv2:
    path: "/path/to/your/ms1m-retinaface-t1"  # Update this!
    
# InsightFace model path (usually auto-detected)
model:
  use_insightface: true
  insightface_model: "buffalo_l"
  # If model is in custom location:
  # insightface_root: "/path/to/.insightface"
```

## Step 4: Install Dependencies

### Check Existing Packages
```python
# Run in a Jupyter cell or terminal
import sys
!{sys.executable} -m pip list | grep -E "(torch|insightface|numpy|yaml)"
```

### Install Missing Dependencies
```bash
# In terminal or Jupyter cell:
pip install -r requirements.txt

# Or install individually:
pip install torch torchvision
pip install insightface
pip install pyyaml numpy scipy
pip install matplotlib seaborn
pip install tqdm
```

## Step 5: Verify InsightFace Model

```python
# Test in a Jupyter cell
import insightface
from insightface.app import FaceAnalysis

# Check if buffalo_l is available
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
print("✓ InsightFace buffalo_l model loaded successfully!")
```

## Step 6: Verify Dataset Path

```python
# Test dataset loading
import os
from src.data.dataset import MS1MV2Dataset

dataset_path = "/path/to/your/ms1m-retinaface-t1"  # Your actual path
if os.path.exists(dataset_path):
    print(f"✓ Dataset found at: {dataset_path}")
    # Try loading a sample
    dataset = MS1MV2Dataset(dataset_path, is_training=False)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
else:
    print(f"✗ Dataset not found at: {dataset_path}")
    print("Please update config.yaml with correct path")
```

## Step 7: Run the Linearizer

### Option A: Using Jupyter Notebook (Recommended for First Run)

1. Open `notebooks/03_linearization.ipynb`
2. Update the config path in the first cell if needed:
   ```python
   # If config.yaml is in parent directory
   with open('../config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   # Or if in same directory:
   with open('config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   ```
3. Run cells sequentially
4. Monitor training progress

### Option B: Using Training Script

```bash
# In terminal on server:
cd /path/to/fast-unlearning-face-recognition
python scripts/train_linearizer.py --config config.yaml

# With custom checkpoint directory:
python scripts/train_linearizer.py --config config.yaml --checkpoint-dir ./checkpoints
```

## Step 8: Monitor Training

Training will:
- Print progress every 10 epochs
- Save checkpoint to `checkpoints/linearizer/linearizer_final.pth`
- Show loss values

Expected output:
```
Using device: cuda
Loading face recognition model...
Original model loaded
Creating Linearizer with sandwich architecture...
Training Linearizer for 100 epochs...
Total parameters: X,XXX,XXX
Epoch 10/100, Loss: 0.XXXXXX
Epoch 20/100, Loss: 0.XXXXXX
...
Training completed!
Saved checkpoint to ./checkpoints/linearizer/linearizer_final.pth
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Add project root to Python path
```python
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
```

### Issue: "InsightFace model not found"
**Solution**: Check model location
```python
import insightface
print(insightface.model_zoo.get_model_list())
# Download if needed:
# insightface.model_zoo.download('buffalo_l')
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `config.yaml`
```yaml
linearizer:
  batch_size: 32  # Reduce from 64
```

### Issue: "Dataset path not found"
**Solution**: Update `config.yaml` with absolute path
```yaml
data:
  ms1mv2:
    path: "/absolute/path/to/ms1m-retinaface-t1"
```

### Issue: "Permission denied"
**Solution**: Check file permissions
```bash
chmod +x scripts/train_linearizer.py
```

## Quick Start Checklist

- [ ] Files transferred to server
- [ ] `config.yaml` updated with correct paths
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] InsightFace model verified
- [ ] Dataset path verified
- [ ] Run `notebooks/03_linearization.ipynb` or `scripts/train_linearizer.py`
- [ ] Training started successfully
- [ ] Checkpoints saved

## Next Steps After Training

1. **Evaluate**: Use `notebooks/05_evaluation.ipynb` to test on benchmarks
2. **Unlearning**: Use trained linearizer for fast unlearning experiments
3. **Visualization**: Check reconstruction quality in notebook

## Support

If you encounter issues:
1. Check error messages carefully
2. Verify all paths in `config.yaml`
3. Ensure all dependencies are installed
4. Check GPU availability: `torch.cuda.is_available()`
