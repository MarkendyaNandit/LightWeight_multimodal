"""
validate.py — Validation and Thresholding Script.

This script serves two purposes:
1. Validates a trained RGBFeatureExtractor on the held-out validation set.
2. Computes the final "anomaly threshold" based on the maximum distance
   of normal validation images to the normal feature center.

How Anomaly Scoring Works:
    During training, the model learned a "center" vector (the average
    normal cookie feature) and pulled all normal features toward it.
    
    Anomaly Score = L2_Distance( feature_vector, center )

    If an image's score > Threshold, it is classified as Anomalous.
    We set the Threshold to be slightly higher than the maximum score
    seen in the normal validation set.

Usage:
    python validate.py
    python validate.py --checkpoint outputs/checkpoints/best_model.pth

Outputs:
    outputs/features/validation_stats.json (Contains center and threshold)
"""

import os
import json
import argparse
import logging
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from config import cfg
from transforms import get_eval_transforms
from dataset import CookieTrainDataset
from feature_head import RGBFeatureExtractor
from train import AnomalyFeatureLoss

# Set up logging for standalone execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_threshold(
    model: torch.nn.Module,
    dataloader: DataLoader,
    center: torch.Tensor,
    device: str,
) -> Dict[str, Any]:
    """
    Run validation data through the model and compute the threshold.

    Args:
        model:      Trained RGBFeatureExtractor.
        dataloader: Validation dataloader (normal images only).
        center:     The learned center vector (1, 256).
        device:     "cpu" or "cuda".

    Returns:
        Dict containing validation statistics and the computed threshold.
    """
    model.eval()
    
    all_distances = []
    
    logger.info("Extracting features for validation set...")
    with torch.no_grad():
        for (images, paths) in dataloader:
            images = images.to(device)
            features = model(images)
            
            # Anomaly score is the squared L2 distance to the center
            # (same metric used in Compactness Loss during training)
            distances = ((features - center.to(device)) ** 2).sum(dim=1)
            all_distances.append(distances.cpu())

    all_distances = torch.cat(all_distances, dim=0)
    
    mean_dist = all_distances.mean().item()
    std_dist = all_distances.std().item() if len(all_distances) > 1 else 0.0
    max_dist = all_distances.max().item()
    
    # The threshold is set to the maximum distance seen in the normal
    # validation set, plus a small margin.
    # Alternatively: mean + 3 * std
    margin = 0.1 * max_dist
    threshold = max_dist + margin

    stats = {
        "num_samples": len(all_distances),
        "mean_distance": mean_dist,
        "std_distance": std_dist,
        "max_distance": max_dist,
        "margin": margin,
        "threshold": threshold,
    }
    
    return stats


def main(args: argparse.Namespace = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="Validate RGB Encoder")
        parser.add_argument(
            "--checkpoint", 
            type=str, 
            default=os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"),
            help="Path to the trained model checkpoint."
        )
        args = parser.parse_args()

    device = cfg.get_device()
    logger.info("=" * 65)
    logger.info("  RGB ENCODER MODULE — VALIDATION & THRESHOLDING")
    logger.info("=" * 65)
    
    if not os.path.isfile(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Please run train.py first to generate a model.")
        return

    # --- Load Model and Loss Function State ---
    model = RGBFeatureExtractor().to(device)
    loss_fn = AnomalyFeatureLoss().to(device)
    
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
    
    epoch = checkpoint["epoch"]
    center = loss_fn.center.detach().cpu()  # (1, 256)
    
    logger.info(f"Model loaded (trained for {epoch} epochs).")
    
    # --- Load Validation Dataset ---
    val_dataset = CookieTrainDataset(
        image_dir=cfg.get_val_dir(),
        transform=get_eval_transforms(),
        return_path=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    
    # --- Compute Threshold ---
    stats = compute_threshold(model, val_loader, center, device)
    
    logger.info("\n--- Validation Statistics ---")
    logger.info(f"  Samples evaluated : {stats['num_samples']}")
    logger.info(f"  Mean distance     : {stats['mean_distance']:.4f}")
    logger.info(f"  Std distance      : {stats['std_distance']:.4f}")
    logger.info(f"  Max distance      : {stats['max_distance']:.4f}")
    logger.info(f"  Margin applied    : +{stats['margin']:.4f}")
    logger.info(f"  =====================================")
    logger.info(f"  ANOMALY THRESHOLD : {stats['threshold']:.4f}")
    logger.info(f"  =====================================")
    
    # --- Save Statistics for Testing ---
    # We save the center vector and the computed threshold so that test.py
    # can load them for inference.
    output_data = {
        "checkpoint_epoch": epoch,
        "validation_stats": stats,
        "anomaly_threshold": stats["threshold"],
        "center_vector": center.squeeze().tolist(),  # List of 256 floats
    }
    
    os.makedirs(cfg.FEATURE_DIR, exist_ok=True)
    stats_path = os.path.join(cfg.FEATURE_DIR, "validation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    logger.info(f"\nValidation stats and center vector saved to:\n  {stats_path}")
    logger.info("\nYou can now proceed to run test.py for final evaluation.")


if __name__ == "__main__":
    main()
