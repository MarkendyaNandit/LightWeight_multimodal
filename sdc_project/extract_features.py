"""
extract_features.py — Feature Extraction Script for the RGB Encoder Module.

After training, this script extracts the 256-dimensional RGB feature vector
from every image in the dataset (train, validation, and test splits).
These features are saved as NumPy .npy files for downstream use.

Outputs:
    outputs/features/train_features.npy      (N_train, 256) float32
    outputs/features/train_labels.npy         (N_train,) int64
    outputs/features/train_paths.npy          (N_train,) str

    outputs/features/val_features.npy         (N_val, 256) float32
    outputs/features/val_labels.npy           (N_val,) int64
    outputs/features/val_paths.npy            (N_val,) str

    outputs/features/test_features.npy        (N_test, 256) float32
    outputs/features/test_labels.npy          (N_test,) int64
    outputs/features/test_paths.npy           (N_test,) str

Integration Note:
    Your teammate developing the Depth Encoder will produce a similar set
    of feature files. The fusion module (GACM) will load both:
        rgb_features  = np.load("train_features.npy")   # (N, 256)
        depth_features = np.load("depth_features.npy")  # (N, 256)
    and concatenate or fuse them for the final anomaly detection pipeline.

Usage:
    python extract_features.py
    python extract_features.py --checkpoint outputs/checkpoints/best_model.pth
    python extract_features.py --split test    # Extract only test features
"""

import os
import argparse
import logging
import time
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from transforms import get_eval_transforms
from dataset import CookieTrainDataset, CookieTestDataset
from feature_head import RGBFeatureExtractor

# Set up logging for standalone execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# FEATURE EXTRACTION FUNCTIONS
# =========================================================================

@torch.no_grad()
def extract_train_features(
    model: torch.nn.Module,
    image_dir: str,
    device: str,
    batch_size: int = cfg.EVAL_BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from training/validation images (normal only).

    Args:
        model:      Trained RGBFeatureExtractor in eval mode.
        image_dir:  Path to the RGB image directory.
        device:     "cpu" or "cuda".
        batch_size: Batch size for inference.

    Returns:
        features: (N, 256) float32 array of feature vectors.
        labels:   (N,) int64 array, all zeros (normal).
        paths:    List of N image file paths.
    """
    model.eval()

    dataset = CookieTrainDataset(
        image_dir=image_dir,
        transform=get_eval_transforms(),
        return_path=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    all_features = []
    all_paths = []

    for images, paths in dataloader:
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_paths.extend(paths)

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.zeros(len(features_array), dtype=np.int64)  # All normal

    return features_array, labels_array, all_paths


@torch.no_grad()
def extract_test_features(
    model: torch.nn.Module,
    test_dir: str,
    device: str,
    batch_size: int = cfg.EVAL_BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from test images (normal + anomalous).

    Args:
        model:      Trained RGBFeatureExtractor in eval mode.
        test_dir:   Path to the test root directory.
        device:     "cpu" or "cuda".
        batch_size: Batch size for inference.

    Returns:
        features: (N, 256) float32 array of feature vectors.
        labels:   (N,) int64 array, 0=normal, 1=anomalous.
        paths:    List of N image file paths.
    """
    model.eval()

    dataset = CookieTestDataset(
        test_dir=test_dir,
        transform=get_eval_transforms(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    all_features = []
    all_labels = []
    all_paths = []

    for images, labels, masks, paths in dataloader:
        images = images.to(device)
        features = model(images)

        all_features.append(features.cpu().numpy())
        all_labels.extend(labels.tolist())
        all_paths.extend(paths)

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.array(all_labels, dtype=np.int64)

    return features_array, labels_array, all_paths


def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    paths: List[str],
    output_dir: str,
    split_name: str,
) -> None:
    """
    Save extracted features, labels, and paths as .npy files.

    Args:
        features:    (N, 256) feature array.
        labels:      (N,) label array.
        paths:       List of N file paths.
        output_dir:  Directory to save files.
        split_name:  Name prefix (e.g., "train", "val", "test").
    """
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, f"{split_name}_features.npy")
    labels_path = os.path.join(output_dir, f"{split_name}_labels.npy")
    paths_path = os.path.join(output_dir, f"{split_name}_paths.npy")

    np.save(features_path, features)
    np.save(labels_path, labels)
    np.save(paths_path, np.array(paths, dtype=object))

    logger.info(
        f"  {split_name}: features {features.shape} saved to {features_path}"
    )
    logger.info(f"  {split_name}: labels {labels.shape} saved to {labels_path}")
    logger.info(f"  {split_name}: paths ({len(paths)}) saved to {paths_path}")


def verify_features(
    features: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> None:
    """
    Verify extracted features meet the interface contract.

    Checks:
        1. Feature dimension is exactly 256.
        2. Features are L2-normalized (norms ≈ 1.0).
        3. Feature statistics are reasonable (no NaN, Inf).
        4. Label values are valid (0 or 1).
    """
    logger.info(f"\n  --- Verification: {split_name} ---")

    # Check shape
    assert features.shape[1] == cfg.FEATURE_DIM, (
        f"Expected 256-dim features, got {features.shape[1]}"
    )
    logger.info(f"  [PASS] Feature dimension: {features.shape[1]}")

    # Check for NaN/Inf
    assert not np.isnan(features).any(), "Features contain NaN!"
    assert not np.isinf(features).any(), "Features contain Inf!"
    logger.info(f"  [PASS] No NaN or Inf values")

    # Check L2 normalization
    norms = np.linalg.norm(features, axis=1)
    mean_norm = norms.mean()
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"Features are not L2-normalized! Mean norm: {mean_norm:.4f}"
    )
    logger.info(f"  [PASS] L2-normalized (mean norm: {mean_norm:.6f})")

    # Check labels
    unique_labels = set(labels.tolist())
    assert unique_labels.issubset({0, 1}), (
        f"Invalid labels found: {unique_labels}"
    )
    normal_count = (labels == 0).sum()
    anomaly_count = (labels == 1).sum()
    logger.info(
        f"  [PASS] Labels: {normal_count} normal, {anomaly_count} anomalous"
    )

    # Feature statistics
    logger.info(f"  Feature stats:")
    logger.info(f"    Mean: {features.mean():.6f}")
    logger.info(f"    Std:  {features.std():.6f}")
    logger.info(f"    Min:  {features.min():.6f}")
    logger.info(f"    Max:  {features.max():.6f}")


# =========================================================================
# MAIN
# =========================================================================

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            description="Extract 256-dim RGB features"
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"),
            help="Path to the trained model checkpoint.",
        )
        parser.add_argument(
            "--split",
            type=str,
            default="all",
            choices=["all", "train", "val", "test"],
            help="Which split(s) to extract features for.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=cfg.FEATURE_DIR,
            help="Directory to save extracted features.",
        )
        args = parser.parse_args()

    device = cfg.get_device()
    logger.info("=" * 65)
    logger.info("  RGB ENCODER MODULE — FEATURE EXTRACTION")
    logger.info("=" * 65)

    # --- Load Model ---
    if not os.path.isfile(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Please run train.py first.")
        return

    logger.info(f"Loading model from: {args.checkpoint}")
    model = RGBFeatureExtractor().to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    logger.info(f"Model loaded (epoch {epoch}).")
    logger.info(f"Output directory: {args.output_dir}")

    start_time = time.time()

    # --- Extract Train Features ---
    if args.split in ("all", "train"):
        logger.info("\n--- Extracting TRAIN features ---")
        train_features, train_labels, train_paths = extract_train_features(
            model, cfg.get_train_dir(), device
        )
        save_features(
            train_features, train_labels, train_paths,
            args.output_dir, "train"
        )
        verify_features(train_features, train_labels, "train")

    # --- Extract Validation Features ---
    if args.split in ("all", "val"):
        logger.info("\n--- Extracting VALIDATION features ---")
        val_features, val_labels, val_paths = extract_train_features(
            model, cfg.get_val_dir(), device
        )
        save_features(
            val_features, val_labels, val_paths,
            args.output_dir, "val"
        )
        verify_features(val_features, val_labels, "val")

    # --- Extract Test Features ---
    if args.split in ("all", "test"):
        logger.info("\n--- Extracting TEST features ---")
        test_features, test_labels, test_paths = extract_test_features(
            model, cfg.get_test_dir(), device
        )
        save_features(
            test_features, test_labels, test_paths,
            args.output_dir, "test"
        )
        verify_features(test_features, test_labels, "test")

    # --- Summary ---
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 65}")
    logger.info(f"  FEATURE EXTRACTION COMPLETE")
    logger.info(f"{'=' * 65}")
    logger.info(f"  Time elapsed:  {elapsed:.1f}s")
    logger.info(f"  Output dir:    {args.output_dir}")
    logger.info(f"  Feature dim:   {cfg.FEATURE_DIM}")
    logger.info(f"")
    logger.info(f"  Files generated:")

    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".npy"):
            fpath = os.path.join(args.output_dir, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            logger.info(f"    {f:35s}  ({size_mb:.2f} MB)")

    logger.info(f"\n  These .npy files are ready for integration with the")
    logger.info(f"  Depth module and the multimodal fusion pipeline.")
    logger.info(f"{'=' * 65}")

    # Cleanup
    model.encoder.remove_hooks()


if __name__ == "__main__":
    main()
