"""
test.py — Final Evaluation Script for the RGB Encoder.

Evaluates the trained model on the complete test set, which contains
both normal images and 4 types of defective images (crack, contamination,
hole, combined).

Process:
    1. Loads the best model weights.
    2. Loads the normal feature center and anomaly threshold calculated
       by validate.py.
    3. Extracts 256-dim features for all 131 test images.
    4. Computes the anomaly score = ||feature - center||^2.
    5. Classifies images: score > threshold -> Anomalous (1)
                          score <= threshold -> Normal (0)
    6. Computes standard anomaly detection metrics:
       - Image-level AUROC (Area Under the ROC Curve)
       - Accuracy, Precision, Recall, F1-Score
       - Defect-specific breakdown

Integration Note:
    This evaluates ONLY the RGB modality. When your teammates finish
    their modules, you will evaluate the fused (RGB + Depth) features.
    The performance here is a baseline showing how much anomaly signal
    can be extracted from RGB alone.

Usage:
    python test.py
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, List
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from config import cfg
from transforms import get_eval_transforms
from dataset import CookieTestDataset
from feature_head import RGBFeatureExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_test_set(
    model: torch.nn.Module,
    dataloader: DataLoader,
    center: torch.Tensor,
    threshold: float,
    device: str,
) -> Dict[str, Any]:
    """
    Run evaluation on the test set.

    Args:
        model:      Trained RGBFeatureExtractor.
        dataloader: CookieTestDataset dataloader.
        center:     Normal feature center (1, 256).
        threshold:  Anomaly score threshold.
        device:     "cpu" or "cuda".

    Returns:
        Dict containing performance metrics and per-image results.
    """
    model.eval()

    all_scores = []
    all_labels = []
    all_preds = []
    all_paths = []

    logger.info(f"Extracting features for {len(dataloader.dataset)} test images...")
    
    with torch.no_grad():
        for batch_idx, (images, labels, masks, paths) in enumerate(dataloader):
            images = images.to(device)
            
            # 1. Extract 256-dim features
            features = model(images)
            
            # 2. Compute anomaly score (squared L2 distance to center)
            distances = ((features - center.to(device)) ** 2).sum(dim=1)
            scores = distances.cpu().numpy()
            
            # 3. Classify based on threshold
            preds = (scores > threshold).astype(int)
            
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_paths.extend(paths)

    # --- Compute Metrics ---
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred = np.array(all_preds)

    # AUROC (Area Under Receiver Operating Characteristic Curve)
    # This is threshold-independent and the most important metric in AD.
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        # Happens if test set only contains one class (e.g., debugging)
        auroc = 0.0

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # --- Breakdown by Defect Type ---
    defect_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for label, pred, path in zip(y_true, y_pred, all_paths):
        # Extract defect type from path: .../test/<defect_type>/rgb/...
        parts = path.replace("\\", "/").split("/")
        try:
            test_idx = parts.index("test")
            defect_type = parts[test_idx + 1]
        except (ValueError, IndexError):
            defect_type = "unknown"
            
        defect_stats[defect_type]["total"] += 1
        if label == pred:
            defect_stats[defect_type]["correct"] += 1

    # Format defect breakdown
    breakdown = {}
    for dtype, counts in defect_stats.items():
        acc = counts["correct"] / max(counts["total"], 1)
        breakdown[dtype] = {
            "accuracy": acc,
            "correct": counts["correct"],
            "total": counts["total"]
        }

    metrics = {
        "auroc": auroc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "defect_breakdown": breakdown,
        "threshold_used": threshold,
    }

    # Save detailed per-image results for error analysis
    per_image_results = [
        {
            "path": p,
            "label": l,
            "score": s,
            "prediction": pr,
            "is_correct": bool(l == pr)
        }
        for p, l, s, pr in zip(all_paths, all_labels, all_scores, all_preds)
    ]

    return metrics, per_image_results


def main(args: argparse.Namespace = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="Evaluate RGB Encoder")
        parser.add_argument(
            "--checkpoint", 
            type=str, 
            default=os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"),
            help="Path to trained model checkpoint."
        )
        parser.add_argument(
            "--stats", 
            type=str, 
            default=os.path.join(cfg.FEATURE_DIR, "validation_stats.json"),
            help="Path to validation stats containing center and threshold."
        )
        args = parser.parse_args()

    device = cfg.get_device()
    logger.info("=" * 65)
    logger.info("  RGB ENCODER MODULE — TEST EVALUATION")
    logger.info("=" * 65)

    if not os.path.isfile(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
        
    if not os.path.isfile(args.stats):
        logger.error(f"Validation stats not found: {args.stats}")
        logger.error("Run validate.py first to compute the threshold.")
        return

    # --- Load Stats (Center & Threshold) ---
    with open(args.stats, "r") as f:
        stats_data = json.load(f)
        
    center_list = stats_data["center_vector"]
    threshold = stats_data["anomaly_threshold"]
    
    # Convert center back to tensor (1, 256)
    center = torch.tensor(center_list, dtype=torch.float32).unsqueeze(0)
    
    logger.info(f"Loaded validation stats:")
    logger.info(f"  Center vector shape: {center.shape}")
    logger.info(f"  Anomaly threshold:   {threshold:.4f}")

    # --- Load Model ---
    model = RGBFeatureExtractor().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model weights from epoch {checkpoint['epoch']}.")

    # --- Load Test Dataset ---
    test_dataset = CookieTestDataset(
        test_dir=cfg.get_test_dir(),
        transform=get_eval_transforms(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    # --- Run Evaluation ---
    metrics, image_results = evaluate_test_set(
        model, test_loader, center, threshold, device
    )

    # --- Log Results ---
    logger.info("\n=======================================================")
    logger.info("  TEST SET METRICS (IMAGE-LEVEL)")
    logger.info("=======================================================")
    logger.info(f"  AUROC     : {metrics['auroc']:.4f}")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score  : {metrics['f1_score']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info("-------------------------------------------------------")
    logger.info("  ACCURACY BY DEFECT TYPE")
    logger.info("-------------------------------------------------------")
    for dtype, stat in metrics["defect_breakdown"].items():
        logger.info(
            f"  {dtype:15s}: {stat['accuracy'] * 100:>5.1f}% "
            f"({stat['correct']:>2d}/{stat['total']:>2d})"
        )
    logger.info("=======================================================\n")

    # --- Save Results ---
    output_data = {
        "metrics": metrics,
        "per_image_results": image_results
    }
    
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    results_path = os.path.join(cfg.LOG_DIR, f"{cfg.EXPERIMENT_NAME}_test_results.json")
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    logger.info(f"Detailed test results saved to:\n  {results_path}")


if __name__ == "__main__":
    main()
