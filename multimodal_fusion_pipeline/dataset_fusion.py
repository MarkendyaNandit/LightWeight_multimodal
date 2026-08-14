"""
dataset_fusion.py — Dataset Loader for Pre-extracted Features.

Loads pre-extracted RGB (256-D) and Depth (256-D) feature arrays and pairs them
for training, validation, and testing of the GACM + Fusion pipeline.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import cfg

class FeaturePairDataset(Dataset):
    """
    Dataset wrapping paired (RGB, Depth) feature vectors.
    """
    def __init__(self, rgb_features: np.ndarray, depth_features: np.ndarray, labels: np.ndarray = None):
        assert len(rgb_features) == len(depth_features), (
            f"Length mismatch: RGB {len(rgb_features)} vs Depth {len(depth_features)}"
        )
        self.rgb = torch.tensor(rgb_features, dtype=torch.float32)
        self.depth = torch.tensor(depth_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.rgb[idx], self.depth[idx], self.labels[idx]
        return self.rgb[idx], self.depth[idx]

def load_feature_splits():
    """
    Load pre-extracted RGB and Depth feature files for train, val, and test.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load RGB Features
    rgb_train = np.load(os.path.join(cfg.RGB_FEATURE_DIR, "train_features.npy"))
    rgb_val = np.load(os.path.join(cfg.RGB_FEATURE_DIR, "val_features.npy"))
    rgb_test = np.load(os.path.join(cfg.RGB_FEATURE_DIR, "test_features.npy"))
    test_labels = np.load(os.path.join(cfg.RGB_FEATURE_DIR, "test_labels.npy"))

    # 2. Load Depth Features
    depth_train = np.load(os.path.join(cfg.DEPTH_FEATURE_DIR, "train_good", "features.npy"))
    depth_val = np.load(os.path.join(cfg.DEPTH_FEATURE_DIR, "validation_good", "features.npy"))

    # Load test depth features in defect order
    test_good = np.load(os.path.join(cfg.DEPTH_FEATURE_DIR, "test_good", "features.npy"))
    test_contam = np.load(os.path.join(cfg.DEPTH_FEATURE_DIR, "test_contamination", "features.npy"))
    test_crack = np.load(os.path.join(cfg.DEPTH_FEATURE_DIR, "test_crack", "features.npy"))
    test_hole = np.load(os.path.join(cfg.DEPTH_FEATURE_DIR, "test_hole", "features.npy"))
    test_comb = np.load(os.path.join(cfg.DEPTH_FEATURE_DIR, "test_combined", "features.npy"))

    depth_test = np.concatenate([test_good, test_contam, test_crack, test_hole, test_comb], axis=0)

    # 3. Build Datasets
    train_ds = FeaturePairDataset(rgb_train, depth_train)
    val_ds = FeaturePairDataset(rgb_val, depth_val)
    test_ds = FeaturePairDataset(rgb_test, depth_test, test_labels)

    # 4. Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
