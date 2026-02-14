# Deepfake Detector - CNN + ViT Hybrid Model

This project uses a hybrid model combining ResNet (CNN) and Vision Transformer (ViT) for deepfake detection.

## Project Structure
```
deepfake-detector/
├── model/
│   └── vit_cnn_model.py       # Hybrid ResNet + ViT model
├── data/
│   └── data.py                # Data loading and transforms
├── train/
│   └── train.py               # Training script
├── Evaluvate/
│   └── Evaluvate_module.py    # Evaluation metrics
├── dataset/                   # Full dataset (24k images)
│   ├── train/
│   ├── valid/
│   └── test/
└── dataset_small/             # Reduced dataset (10k images)
    ├── train/
    ├── valid/
    └── test/
```

## Model Architecture
- **CNN Backbone**: ResNet50 or ResNet18 (configurable)
- **ViT Backbone**: ViT-B/16
- **Fusion**: Concatenation of CNN and ViT features
- **Output**: Binary classification (fake/real)

## Quick Start

### 1. Train with Both CNN and ViT (Recommended)
```powershell
# Train on small dataset (faster, good for testing)
python train\train.py --data_root dataset_small --epochs 5 --batch_size 8 --num_workers 0

# Train on full dataset
python train\train.py --data_root dataset --epochs 10 --batch_size 8 --num_workers 0
```

### 2. Train with Lightweight ResNet18
```powershell
python train\train.py --data_root dataset_small --epochs 5 --batch_size 8 --cnn_backbone resnet18
```

### 3. Train with Frozen Backbones (Only heads trainable)
```powershell
python train\train.py --data_root dataset_small --epochs 5 --batch_size 8 --freeze_backbones
```

### 4. Train without Pretrained Weights
```powershell
python train\train.py --data_root dataset_small --epochs 10 --batch_size 8 --no_cnn_pretrained --no_vit_pretrained
```

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `dataset` | Path to dataset folder |
| `--batch_size` | `8` | Batch size for training |
| `--num_workers` | `0` | DataLoader workers (use 0 on Windows) |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--cnn_backbone` | `resnet50` | CNN backbone: `resnet50` or `resnet18` |
| `--freeze_backbones` | `False` | Freeze CNN and ViT (only train heads) |
| `--no_cnn_pretrained` | `False` | Don't use pretrained CNN weights |
| `--no_vit_pretrained` | `False` | Don't use pretrained ViT weights |
| `--checkpoint_dir` | `checkpoints` | Where to save checkpoints |
| `--log_dir` | `runs` | TensorBoard log directory |
| `--resume` | `` | Resume from checkpoint path |

## Dataset Information

### Full Dataset (`dataset/`)
- Train: 20,000 images (10k fake, 10k real)
- Valid: 2,002 images (1001 fake, 1001 real)
- Test: 2,000 images (1000 fake, 1000 real)
- Total: 24,002 images

### Small Dataset (`dataset_small/`)
- Train: 8,000 images (4k fake, 4k real)
- Valid: 1,000 images (500 fake, 500 real)
- Test: 1,000 images (500 fake, 500 real)
- Total: 10,000 images

## Model Configuration

### Default (Both CNN and ViT Pretrained & Trainable)
✅ This is the **recommended** configuration
- Both CNN and ViT use ImageNet pretrained weights
- All parameters are trainable (backbones + heads)
- Best performance for deepfake detection

```python
model = HybridModel(
    cnn_backbone='resnet50',
    cnn_pretrained=True,      # ✓ Use pretrained CNN
    vit_pretrained=True,      # ✓ Use pretrained ViT
    freeze_backbones=False    # ✓ Train both backbones
)
```

### Parameter Counts
- ResNet50 + ViT: ~152M total, ~152M trainable
- ResNet18 + ViT: ~97M total, ~97M trainable
- Frozen backbones: ~152M total, ~1M trainable (heads only)

## Monitoring Training

### View TensorBoard
```powershell
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```
Then open: http://127.0.0.1:6006

### Checkpoints
- Saved after each epoch: `checkpoints/epoch_XXX_auc_Y.YYYY.pth`
- Best model: `checkpoints/best.pth`

## Testing

### Run Test Script
```powershell
python test_setup.py
```
This validates:
- Model creation with CNN + ViT
- Forward pass
- Dataloader functionality

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.0.0
tensorboard>=2.10.0
pillow>=9.0.0
numpy>=1.21.0
```

Install:
```powershell
pip install torch torchvision scikit-learn tensorboard pillow numpy
```

## Performance Tips

### For CPU Training
- Use `--cnn_backbone resnet18` (lighter)
- Use `--batch_size 4` or `--batch_size 8`
- Use `--num_workers 0`
- Consider `--freeze_backbones` for faster training

### For GPU Training
- Use `--cnn_backbone resnet50`
- Use `--batch_size 32` or higher
- Use `--num_workers 4`
- Add `--use_amp` for mixed precision

## Example Training Session

```powershell
# Navigate to project
cd C:\Users\suraj\OneDrive\Documents\deepfake-detector

# Train with both CNN and ViT (pretrained, trainable)
python train\train.py --data_root dataset_small --epochs 5 --batch_size 8

# Output will show:
# Using device: cpu
# Classes: {'fake': 0, 'real': 1}
# Model: CNN=resnet50, CNN_pretrained=True, ViT_pretrained=True
# Freeze_backbones=False
# Total parameters: 152,XXX,XXX
# Trainable parameters: 152,XXX,XXX
# Epoch 0/4  train_loss=0.XXXX  val_loss=0.XXXX  val_auc=0.XXXX  time=XXXs
```

## Troubleshooting

### "No module named 'model'"
Run from project root: `cd C:\Users\suraj\OneDrive\Documents\deepfake-detector`

### "FileNotFoundError: dataset_small\train"
Create dataset_small or use `--data_root dataset`

### Slow training on CPU
- Use `--cnn_backbone resnet18`
- Use `--batch_size 4`
- Use `--freeze_backbones`

### Out of memory
- Reduce `--batch_size`
- Use `--cnn_backbone resnet18`
- Use `--freeze_backbones`

## Verification

To confirm both CNN and ViT are included and trainable:
```powershell
python -B - <<'PY'
from model.vit_cnn_model import HybridModel
m = HybridModel(cnn_pretrained=True, vit_pretrained=True, freeze_backbones=False)
print(f"Trainable: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
print(f"Total: {sum(p.numel() for p in m.parameters()):,}")
PY
```

Both numbers should be equal (all parameters trainable).
