# Dataset Information

## Overview
This document contains comprehensive statistics and details about the datasets used for training the deepfake detection model.

---

## Full Dataset (`dataset/`)

### Dataset Split
| Split | Fake Images | Real Images | Total Images |
|-------|-------------|-------------|--------------|
| Train | 10,000 | 10,000 | 20,000 |
| Validation | 1,001 | 1,001 | 2,002 |
| Test | 1,000 | 1,000 | 2,000 |
| **Total** | **12,001** | **12,001** | **24,002** |

### Directory Structure
```
dataset/
├── train/
│   ├── fake/     (10,000 images)
│   └── real/     (10,000 images)
├── val/
│   ├── fake/     (1,001 images)
│   └── real/     (1,001 images)
└── test/
    ├── fake/     (1,000 images)
    └── real/     (1,000 images)
```

---

## Small Dataset (`dataset_small/`)

### Dataset Split
| Split | Fake Images | Real Images | Total Images |
|-------|-------------|-------------|--------------|
| Train | 4,000 | 4,000 | 8,000 |
| Validation | 500 | 500 | 1,000 |
| Test | 500 | 500 | 1,000 |
| **Total** | **5,000** | **5,000** | **10,000** |

### Directory Structure
```
dataset_small/
├── train/
│   ├── fake/     (4,000 images)
│   └── real/     (4,000 images)
├── val/
│   ├── fake/     (500 images)
│   └── real/     (500 images)
└── test/
    ├── fake/     (500 images)
    └── real/     (500 images)
```

---

## Data Characteristics

### Image Specifications
- **Image Size**: 224 × 224 pixels
- **Format**: Standard image formats (JPG, PNG)
- **Color Space**: RGB (3 channels)

### Class Distribution
- **Balanced Dataset**: 50% fake, 50% real for all splits
- **No Class Imbalance**: Equal representation ensures fair training

### Data Augmentation

#### Training Data
```python
Augmentations:
- transforms.RandomResizedCrop(224)
- transforms.RandomHorizontalFlip()
- transforms.ToTensor()
- transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
```

#### Validation & Test Data
```python
Augmentations:
- transforms.Resize(252)
- transforms.CenterCrop(224)
- transforms.ToTensor()
- transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
```

### Normalization
- **Mean**: [0.485, 0.456, 0.406]
- **Std Dev**: [0.229, 0.224, 0.225]
- **Reason**: ImageNet normalization for compatibility with pretrained models

---

## Batch Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Num Workers | 0 (Windows) |
| Pin Memory | True |
| Shuffle (Train) | True |
| Shuffle (Val/Test) | False |

---

## Data Processing Pipeline

### Training Loader
```python
get_dataloaders(root_dir='dataset_small',
                batch_size=8,
                num_workers=0,
                img_size=224,
                pin_memory=True)
```

**Output**: (train_loader, val_loader, test_loader, class_to_idx)

### Class Mapping
```python
class_to_idx = {'fake': 0, 'real': 1}
```

---

## Dataset Usage

### For Quick Testing
Use `dataset_small/` (10,000 images)
- Faster training loops
- Good for development and debugging
- Approximately 1 hour per epoch on CPU

### For Production Training
Use `dataset/` (24,000 images)
- Better model generalization
- More comprehensive training data
- Approximately 2.5 hours per epoch on CPU

---

## Notes
- All datasets are pre-split into train/val/test
- Balanced class distribution ensures unbiased training
- ImageNet normalization is applied for pretrained model compatibility
- Data augmentation only applied to training set to prevent information leakage
