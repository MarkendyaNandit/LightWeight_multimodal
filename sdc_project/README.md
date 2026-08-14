# RGB Encoder Module — MobileNetV3 Feature Extractor

## Overview

This module implements the **RGB Encoder** component of the multimodal industrial anomaly detection system based on the paper *"Text-Guided Multimodal Unified Industrial Anomaly Detection"* (arXiv: 2604.22899).

The RGB Encoder takes cookie images from the MVTec 3D-AD dataset, fine-tunes an ImageNet-pretrained MobileNetV3-Large network, and produces a **256-dimensional feature vector** for each image. This feature vector captures the visual appearance of normal cookies, enabling anomaly detection through distance-based scoring.

### Architecture

```
RGB Image (3, 800, 800)
        │
        ▼
    Resize to (3, 224, 224)
    ImageNet Normalization
        │
        ▼
    MobileNetV3-Large (Pretrained)
    ┌───────────────────────────────────────┐
    │  Blocks 0-8:  FROZEN (ImageNet)       │
    │  Block 7:     TAP → 80ch (14×14)      │
    │  Blocks 9-16: FINE-TUNED              │
    │  Block 11:    TAP → 112ch (14×14)     │
    │  Block 13:    TAP → 160ch (7×7)       │
    └───────────────────────────────────────┘
        │
        ▼
    Global Average Pooling at each TAP
    Concatenate: [80 + 112 + 160] = 352-dim
        │
        ▼
    Feature Head (MLP)
    ┌────────────────────────────────┐
    │  Linear(352, 512)             │
    │  LayerNorm(512)               │
    │  GELU                         │
    │  Dropout(0.1)                 │
    │  Linear(512, 256)             │
    │  L2 Normalize                 │
    └────────────────────────────────┘
        │
        ▼
    256-dim RGB Feature Vector (unit length)
```

## Project Structure

```
sdc_project/
├── config.py              # Central configuration (paths, hyperparameters)
├── transforms.py          # Image preprocessing and augmentation pipelines
├── dataset.py             # Custom PyTorch Dataset classes for MVTec 3D-AD
├── mobilenet_encoder.py   # MobileNetV3-Large backbone with multi-scale extraction
├── feature_head.py        # Projection MLP + RGBFeatureExtractor wrapper
├── train.py               # Self-supervised training loop
├── validate.py            # Validation and anomaly threshold computation
├── test.py                # Final evaluation with AUROC, accuracy, F1, etc.
├── extract_features.py    # Save 256-dim features as .npy files
├── utils.py               # Shared utility functions
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── outputs/
    ├── checkpoints/       # Saved model weights
    │   ├── best_model.pth
    │   └── final_model.pth
    ├── logs/              # Training logs and test results
    ├── features/          # Extracted .npy feature files
    └── visualizations/    # t-SNE plots and other visuals
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the **Cookie** category from [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) and place it so the path resolves to:

```
sdc project 3rd year/
├── cookie/cookie/
│   ├── train/good/rgb/    (210 images)
│   ├── validation/good/rgb/ (22 images)
│   └── test/
│       ├── good/rgb/       (28 images)
│       ├── crack/rgb/      (27 images)
│       ├── contamination/rgb/ (25 images)
│       ├── hole/rgb/       (26 images)
│       └── combined/rgb/   (25 images)
└── sdc_project/            (this code)
```

### 3. Verify Configuration

```bash
python config.py
```

### 4. Train the Model

```bash
python train.py --epochs 100 --batch-size 8 --lr 0.0003
```

### 5. Compute Anomaly Threshold

```bash
python validate.py
```

### 6. Evaluate on Test Set

```bash
python test.py
```

### 7. Extract Features for Fusion

```bash
python extract_features.py
```

## Training Details

### Why Transfer Learning?

With only **210 normal training images**, training a CNN from scratch would severely overfit. ImageNet-pretrained MobileNetV3 provides:
1. **Universal low-level features** (edges, textures) that generalize to any domain.
2. **Mid-level patterns** (shapes, contours) useful for surface inspection.
3. **High-level representations** that we fine-tune for cookie-specific appearance.

### Self-Supervised Loss

Since anomaly detection has **no defect labels during training**, we use:

- **Consistency Loss**: Two augmented views of the same image → features should be identical.
  `L_cos = 1 - cosine_similarity(f(aug_A(x)), f(aug_B(x)))`

- **Compactness Loss**: All normal features → pulled toward a learned center.
  `L_compact = ||f(x) - center||²`

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Encoder | MobileNetV3-Large | 5.4M params, good accuracy/speed trade-off |
| Frozen Layers | 0–8 | Preserves universal ImageNet features |
| Fine-Tuned Layers | 9–16 | Adapts to cookie-specific textures |
| Feature Dim | 256 | Interface contract with Depth module |
| Learning Rate | 3e-4 (head), 3e-5 (encoder) | 10x lower for pretrained layers |
| Optimizer | AdamW | Weight decay prevents forgetting |
| Scheduler | Cosine Annealing | Smooth LR decay for fine-tuning |
| Early Stopping | 15 epochs patience | Prevents overfitting |

## Integration with Other Modules

This module outputs a **256-dim L2-normalized feature vector** per image. Your teammates' modules consume it as follows:

| Module | Owner | Consumes | Purpose |
|--------|-------|----------|---------|
| Depth Encoder | Teammate A | N/A (parallel) | Produces 256-dim depth features |
| GACM | Teammate B | RGB features (256-dim) | Fuses RGB + Depth features |
| Text Encoder (CLIP) | Teammate C | N/A (parallel) | Produces text embeddings |
| OCTA | Teammate D | RGB features (256-dim) | Aligns vision-text features |
| Anomaly Scorer | Teammate E | Fused features | Final anomaly map |

### Interface Contract

```python
# Your teammates can use the trained model like this:
from feature_head import RGBFeatureExtractor

model = RGBFeatureExtractor()
checkpoint = torch.load("outputs/checkpoints/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Extract features
features = model(rgb_images)  # (B, 256) L2-normalized
```

Or load pre-extracted features:

```python
import numpy as np

features = np.load("outputs/features/test_features.npy")  # (131, 256)
labels = np.load("outputs/features/test_labels.npy")       # (131,)
```

## Adding New Categories

To switch from Cookie to another MVTec 3D-AD category (e.g., Bagel):

1. Download the new category dataset.
2. In `config.py`, change:
   ```python
   DATASET_ROOT = "path/to/bagel/bagel"
   CATEGORY = "bagel"
   ```
3. Re-run training: `python train.py`

No other code changes are needed — the module is category-agnostic.

## File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Single source of truth for all paths, hyperparameters, and settings |
| `transforms.py` | Dual-view augmentation (training) and deterministic eval pipeline |
| `dataset.py` | `CookieTrainDataset` (normal only) and `CookieTestDataset` (all defects) |
| `mobilenet_encoder.py` | MobileNetV3-Large backbone with multi-scale feature hooks |
| `feature_head.py` | 2-layer MLP projection (352→512→256) with L2 normalization |
| `train.py` | Full training pipeline with early stopping and checkpointing |
| `validate.py` | Computes anomaly threshold from validation set |
| `test.py` | Final evaluation: AUROC, accuracy, precision, recall, F1 |
| `extract_features.py` | Saves features as .npy files for downstream fusion |
| `utils.py` | Shared utilities: seeding, device, parameter counting, t-SNE |
