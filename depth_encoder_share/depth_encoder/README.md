# Depth MobileNetV3 Encoder

Self-supervised depth feature encoder for the **MVTec 3D-AD** industrial
anomaly detection dataset.  This is **Milestone 1** of a multimodal
anomaly detection framework inspired by *Text-Guided Multimodal Unified
Industrial Anomaly Detection*.

---

## Architecture

```
XYZ TIFF (3-ch float32)
       ↓
DepthPreprocessor
  · Extract Z channel
  · Fill invalid pixels (NaN/0 → median)
  · Normalise to [0,1]
  · Resize to 224×224
  · Repeat to 3 channels
  · ImageNet normalisation
       ↓
DepthAugmentation  (×2 views for VICReg)
       ↓
MobileNetV3-Large backbone (ImageNet-pretrained)
  · Intermediate feature maps tapped at layers [3, 7, 13]
       ↓
AdaptiveAvgPool2d → flatten (960-d)
       ↓
Projection Head:
  Linear(960→512) → BN1d → ReLU → Dropout(0.1) → Linear(512→256)
       ↓
L2 normalise
       ↓
256-dimensional embedding
```

During **training** only: an Expander MLP (256 → 2048) sits on top
for the VICReg loss computation and is discarded at inference.

---

## Self-Supervised Method: VICReg

**VICReg** (Variance-Invariance-Covariance Regularisation, Bardes et al.
ICLR 2022) was chosen over BYOL / DINO / SimSiam because:

| Criterion | VICReg | BYOL | SimSiam | DINO |
|---|:---:|:---:|:---:|:---:|
| No momentum encoder | ✓ | ✗ | ✓ | ✗ |
| No collapse risk | ✓ | ✗ | ✗ | ✓ |
| Small dataset stable | ✓ | ✗ | ✗ | ✗ |
| Simple training | ✓ | — | ✓ | ✗ |

---

## Project Structure

```
depth_encoder/
├── configs/
│   └── config.yaml            ← all hyperparameters
├── data/
│   ├── dataset.py             ← MVTec3DDepthDataset
│   └── transforms.py          ← DepthPreprocessor, DepthAugmentation
├── models/
│   ├── depth_encoder.py       ← MobileNetV3 backbone + projection head
│   └── vicreg.py              ← VICReg SSL wrapper + Expander MLP
├── training/
│   ├── losses.py              ← VICReg loss (inv + var + cov)
│   └── trainer.py             ← training + validation loop
├── utils/
│   ├── checkpointing.py       ← save / load / encoder-only extraction
│   └── logging_utils.py       ← JSON-lines metrics logger
├── scripts/
│   ├── train.py               ← training entry point
│   └── extract_features.py    ← feature extraction entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd c:\Users\cmnan\Documents\anamoly\depth_encoder
pip install -r requirements.txt
```

### 2. Train

```bash
python scripts/train.py
```

Optional overrides:
```bash
python scripts/train.py --epochs 200 --batch-size 16 --lr 1e-3
python scripts/train.py --resume checkpoints/epoch_0050.pt
python scripts/train.py --device cpu
```

### 3. Extract features

```bash
# Train split (default)
python scripts/extract_features.py

# Validation split
python scripts/extract_features.py --split validation/good

# With intermediate feature maps
python scripts/extract_features.py --save-feature-maps

# Custom checkpoint
python scripts/extract_features.py --checkpoint checkpoints/epoch_0100.pt
```

Outputs are saved to `features/<split_name>/`:
```
features/train_good/
├── features.npy       ← (N, 256) float32 embeddings
├── paths.txt          ← file path for each row
└── feature_maps/      ← (optional) per-layer maps
    ├── layer_03.npy
    ├── layer_07.npy
    └── layer_13.npy
```

---

## Configuration

All hyperparameters live in `configs/config.yaml`.
Key settings:

| Section | Key | Default | Notes |
|---|---|---|---|
| `dataset` | `root` | `c:/.../anamoly` | Dataset root |
| `model` | `embedding_dim` | `256` | Final embedding size |
| `model` | `feature_layers` | `[3, 7, 13]` | Intermediate tap points |
| `vicreg` | `sim_coeff` | `25.0` | λ — invariance weight |
| `vicreg` | `std_coeff` | `25.0` | μ — variance weight |
| `vicreg` | `cov_coeff` | `1.0` | ν — covariance weight |
| `training` | `epochs` | `100` | Total training epochs |
| `training` | `batch_size` | `32` | Reduce to 16 on small GPU |
| `training` | `optimizer.lr` | `3e-4` | Base learning rate |

---

## Load Trained Encoder in Other Code

```python
from models.depth_encoder import build_encoder
from utils.checkpointing import load_encoder_only
import yaml, torch

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

encoder = build_encoder(cfg["model"])
load_encoder_only("checkpoints/best.pt", encoder)
encoder.eval()

# Forward pass
x = torch.randn(1, 3, 224, 224)
embedding, feature_maps = encoder(x)
# embedding.shape → (1, 256)
# feature_maps    → {3: Tensor, 7: Tensor, 13: Tensor}
```

---

## Future Integration Points

This encoder is designed to slot into the full multimodal pipeline:

```
RGB MobileNetV3 Encoder  ─┐
                           ├─► GACM (RGB↔Depth alignment)
Depth MobileNetV3 Encoder ─┘         ↓
                                 Unified Features
                                      ↓
CLIP Text Encoder ─────────────► OCTA (cross-modal)
                                      ↓
                               Anomaly Score Map
```

- `DepthEncoder.get_feature_channels()` exposes the channel dims for GACM input
- Checkpoints use `encoder.*` key prefix so RGB encoder can use the same save/load utilities
- `feature_maps` dict keys match the `feature_layers` config — align RGB encoder with the same indices
