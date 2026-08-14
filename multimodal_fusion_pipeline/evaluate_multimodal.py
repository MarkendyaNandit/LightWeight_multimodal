"""
evaluate_multimodal.py — Phase 10: Multimodal Inference & Text Comparison.

Performs the final evaluation step:
1. Passes RGB (256-D) + Depth (256-D) through trained GACM+Fusion network to get 
   the Combined Visual Feature Vector (F_vis).
2. Generates the Object-Conditioned Text Vector (F_p) from Member 1's OCTA pipeline.
3. Compares F_vis against F_p at the final step via Cosine Similarity / Distance.
4. Computes AUROC, Accuracy, Precision, Recall, F1, and per-defect breakdowns.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from collections import defaultdict

# Insert paths to import friend modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "member1_text_pipeline"))

from config import cfg
from fusion_model import VisualFusionPipeline
from dataset_fusion import load_feature_splits
from models.prompt_templates import PromptGenerator
from models.octa import OCTA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def evaluate():
    device = cfg.DEVICE
    logger.info("=" * 65)
    logger.info("  PHASE 10 — MULTIMODAL INFERENCE & FINAL TEXT COMPARISON")
    logger.info("=" * 65)
    logger.info(f"Device: {device}")

    # 1. Load Data
    _, _, test_loader = load_feature_splits()

    # 2. Load Trained Visual Fusion Model
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_fusion_model.pth")
    if not os.path.isfile(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}. Run train_fusion.py first.")
        return

    model = VisualFusionPipeline(dim=cfg.FEATURE_DIM, hidden_dim=cfg.HIDDEN_DIM).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded trained Fusion + GACM model from epoch {checkpoint['epoch']}")

    # 3. Compute Combined Visual Feature Vectors (F_vis) for Test Set
    f_vis_list = []
    labels_list = []

    with torch.no_grad():
        for f_rgb, f_depth, labels in test_loader:
            f_rgb = f_rgb.to(device)
            f_depth = f_depth.to(device)

            f_vis = model(f_rgb, f_depth)
            f_vis_list.append(f_vis.cpu().numpy())
            labels_list.extend(labels.numpy())

    f_vis_test = np.concatenate(f_vis_list, axis=0) # (131, 256)
    y_true = np.array(labels_list)                   # (131,)

    logger.info(f"Extracted Fused Visual Vectors (F_vis): {f_vis_test.shape}")

    # 4. Generate OCTA Text Vector (F_p) for "cookie"
    octa_model = OCTA(dim=cfg.FEATURE_DIM).to(device)
    gen = PromptGenerator(classes=["cookie"])
    prompts = gen.generate_prompts("cookie")

    # Generate mock embeddings for OCTA input
    torch.manual_seed(cfg.SEED)
    mock_clip_embeds = torch.randn(len(prompts), cfg.FEATURE_DIM).to(device)
    with torch.no_grad():
        f_p = octa_model(mock_clip_embeds)                     # (1, 256)
        f_p_norm = F.normalize(f_p, p=2, dim=1).cpu().numpy() # (1, 256)

    logger.info(f"Generated OCTA Text Prototype Vector (F_p): {f_p_norm.shape}")

    # 5. Final Step Comparison: Distance D(F_vis, F_p)
    # Cosine distance = 1 - CosineSimilarity
    cos_sim = np.dot(f_vis_test, f_p_norm.T).squeeze()       # (131,)
    anomaly_scores = 1.0 - cos_sim                          # Higher distance = more anomalous

    # Calculate metrics
    auroc = roc_auc_score(y_true, anomaly_scores)

    # Optimal threshold search
    sorted_scores = sorted(set(anomaly_scores.tolist()))
    best_acc = 0.0
    best_thresh = sorted_scores[0]

    for t in sorted_scores:
        preds = (anomaly_scores > t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    final_preds = (anomaly_scores > best_thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, final_preds, average="binary", zero_division=0
    )

    # Per-defect breakdown
    defect_names = ["good"] * 28 + ["contamination"] * 25 + ["crack"] * 27 + ["hole"] * 26 + ["combined"] * 25
    breakdown = defaultdict(lambda: {"total": 0, "correct": 0})

    for label, pred, dtype in zip(y_true, final_preds, defect_names):
        breakdown[dtype]["total"] += 1
        if label == pred:
            breakdown[dtype]["correct"] += 1

    # Print results
    logger.info("\n=======================================================")
    logger.info("  FINAL MULTIMODAL EVALUATION RESULTS")
    logger.info("=======================================================")
    logger.info(f"  Comparison Method : Fused Visual (F_vis) vs OCTA Text (F_p)")
    logger.info(f"  AUROC             : {auroc:.4f}")
    logger.info(f"  Optimal Threshold : {best_thresh:.6f}")
    logger.info(f"  Accuracy          : {best_acc:.4f} ({best_acc*100:.1f}%)")
    logger.info(f"  Precision         : {precision:.4f}")
    logger.info(f"  Recall            : {recall:.4f}")
    logger.info(f"  F1-Score          : {f1:.4f}")
    logger.info("-------------------------------------------------------")
    logger.info("  ACCURACY BY DEFECT TYPE")
    logger.info("-------------------------------------------------------")
    for dtype in ["good", "contamination", "crack", "hole", "combined"]:
        b = breakdown[dtype]
        acc = b["correct"] / max(b["total"], 1)
        logger.info(f"  {dtype:15s}: {acc * 100:>5.1f}% ({b['correct']:>2d}/{b['total']:>2d})")
    logger.info("=======================================================\n")

    # Save results
    results_path = os.path.join(cfg.LOG_DIR, "multimodal_eval_results.json")
    results_data = {
        "auroc": auroc,
        "optimal_threshold": float(best_thresh),
        "accuracy": float(best_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "defect_breakdown": {k: {"accuracy": v["correct"]/v["total"], "correct": v["correct"], "total": v["total"]} for k, v in breakdown.items()}
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")

if __name__ == "__main__":
    evaluate()
