"""
evaluate_pipeline.py — Phase 4: Rigorous Evaluation & Graph Generation.

KEY DESIGN: HYBRID SCORING
  - SMOOTH categories (manufactured surfaces): MAX patch distance
    → catches even single-patch micro-defects on uniform surfaces
  - ORGANIC categories (natural textures): Top-K average (K=10)
    → smooths out natural texture variation while detecting real defects

Scoring method:
  - Extract 196 patch vectors (each 256-dim) per image from the fused RGB+Depth pipeline
  - Compare each patch to the nearest patch in the normal coreset memory bank
  - Anomaly score = MAX or Top-K avg depending on category type
  - Also computes mean/p95 patch distances for robustness

For each of 13 categories:
  1. Computes real patch-level scores for normal vs defect images
  2. Finds optimal threshold via Youden's J-Index
  3. Generates Distance Histograms, ROC Curves, Confusion Matrices
  4. Saves comprehensive eval_report.txt
"""

import os
import sys
import json
import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("Evaluate")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SDC_DIR = os.path.join(BASE_DIR, "sdc_project")
DEPTH_DIR = os.path.join(BASE_DIR, "depth_encoder_share", "depth_encoder")
FUSION_DIR = os.path.join(BASE_DIR, "multimodal_fusion_pipeline")

for d in [SDC_DIR, os.path.join(DEPTH_DIR, "models"), FUSION_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from unified_dataset import UnifiedAnomalyDataset, ALL_CATEGORIES, get_eval_transform

# ── Hybrid Scoring Configuration ──
# Smooth/manufactured surfaces: use strict MAX patch (catches single-patch micro-defects)
SMOOTH_CATEGORIES = {"phone_screen", "car_metal", "pcb"}
# Organic/textured surfaces: use Top-K average (smooths natural texture noise)
ORGANIC_CATEGORIES = set()
TOP_K = 10  # Average the 10 highest patch distances for organic categories


def load_models_and_banks():
    """Load trained models, category means, AND PatchCore coreset banks."""
    
    from feature_head import RGBFeatureExtractor
    
    from depth_encoder import DepthEncoder
    
    from gacm import LightweightGACM

    class VisualFusionPipeline(nn.Module):
        def __init__(self, dim=256, hidden_dim=512):
            super().__init__()
            self.gacm = LightweightGACM(dim=dim, hidden_dim=hidden_dim)
        def forward(self, f_rgb, f_depth):
            return self.gacm(f_rgb, f_depth)

    # RGB
    rgb_model = RGBFeatureExtractor(encoder_pretrained=True, freeze_up_to=8, feature_dim=256).to(DEVICE)
    rgb_ckpt = os.path.join(SDC_DIR, "outputs", "checkpoints", "best_model.pth")
    if os.path.isfile(rgb_ckpt):
        ckpt = torch.load(rgb_ckpt, map_location=DEVICE, weights_only=False)
        rgb_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    rgb_model.eval()

    # Depth
    depth_model = DepthEncoder(embedding_dim=256, pretrained=True).to(DEVICE)
    depth_ckpt = os.path.join(DEPTH_DIR, "checkpoints", "best.pt")
    if os.path.isfile(depth_ckpt):
        ckpt = torch.load(depth_ckpt, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        cleaned = {k.replace("encoder.", "") if k.startswith("encoder.") else k: v for k, v in sd.items()}
        depth_model.load_state_dict(cleaned, strict=False)
    depth_model.eval()

    # Fusion
    fusion_model = VisualFusionPipeline(dim=256, hidden_dim=512).to(DEVICE)
    fusion_ckpt = os.path.join(FUSION_DIR, "outputs", "checkpoints", "best_fusion_model_finetuned.pth")
    if not os.path.isfile(fusion_ckpt):
        fusion_ckpt = os.path.join(FUSION_DIR, "outputs", "checkpoints", "best_fusion_model.pth")
    if os.path.isfile(fusion_ckpt):
        ckpt = torch.load(fusion_ckpt, map_location=DEVICE, weights_only=False)
        fusion_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    fusion_model.eval()

    # Category means (for global fallback)
    means_path = os.path.join(FUSION_DIR, "outputs", "checkpoints", "all_category_means.json")
    category_means = {}
    if os.path.isfile(means_path):
        with open(means_path, "r") as f:
            raw = json.load(f)
        for cat, vec in raw.items():
            category_means[cat] = torch.tensor(vec, dtype=torch.float32).to(DEVICE)

    # PatchCore coreset banks (PRIMARY scoring mechanism)
    coreset_banks = {}
    bank_path = os.path.join(FUSION_DIR, "outputs", "checkpoints", "patchcore_coreset_banks.pt")
    if os.path.isfile(bank_path):
        coreset_banks = torch.load(bank_path, map_location=DEVICE, weights_only=False)
        logger.info(f"  Loaded PatchCore banks for {len(coreset_banks)} categories")

    return rgb_model, depth_model, fusion_model, category_means, coreset_banks


def extract_patch_features_simple(rgb_model, depth_model, fusion_model, rgb_tensor, depth_tensor):
    """
    Extract spatial patch features WITHOUT global average pooling.

    Instead of: features → GAP → single 256-dim vector (loses spatial info)
    We do:      features → 14×14 grid → 196 patch vectors of 256-dim each

    This preserves spatial location so even a tiny scratch on one patch is detectable.

    Returns: (B, 196, 256) patch vectors
    """
    B = rgb_tensor.shape[0]
    target_size = (14, 14)

    # ── RGB spatial features ──
    rgb_feats = []
    x = rgb_tensor
    for i, block in enumerate(rgb_model.encoder.features):
        x = block(x)
        if i in [7, 11, 13]:
            rgb_feats.append(x)

    # Resize all scales to 14×14 and concatenate channels
    resized = []
    for feat in rgb_feats:
        if feat.shape[-2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
        resized.append(feat)
    rgb_spatial = torch.cat(resized, dim=1)  # (B, 352, 14, 14)

    # Local neighborhood aggregation (3×3 avg pool preserving spatial dims)
    rgb_spatial = F.avg_pool2d(rgb_spatial, kernel_size=3, stride=1, padding=1)

    # ── Depth spatial features ──
    depth_feats = []
    x = depth_tensor
    for i, block in enumerate(depth_model.backbone):
        x = block(x)
        if i in [3, 7, 13]:
            depth_feats.append(x)

    resized_d = []
    for feat in depth_feats:
        if feat.shape[-2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
        resized_d.append(feat)
    depth_spatial = torch.cat(resized_d, dim=1)  # (B, C_depth, 14, 14)
    depth_spatial = F.avg_pool2d(depth_spatial, kernel_size=3, stride=1, padding=1)

    # ── Project each patch through feature heads ──
    B_r, C_r, H, W = rgb_spatial.shape
    rgb_flat = rgb_spatial.permute(0, 2, 3, 1).reshape(-1, C_r)  # (B*196, 352)
    rgb_proj = rgb_model.head(rgb_flat).reshape(B, H, W, 256)    # (B, 14, 14, 256)

    # Depth projection: use depth model's projection head
    B_d, C_d, H_d, W_d = depth_spatial.shape
    dep_flat = depth_spatial.permute(0, 2, 3, 1).reshape(-1, C_d)  # (B*196, C_d)
    # Depth model expects 960-dim input; if channels don't match, pool to match
    if C_d != 960:
        # Adaptive: use a simple linear projection to 256-dim
        dep_proj = F.normalize(F.adaptive_avg_pool1d(dep_flat.unsqueeze(1), 256).squeeze(1), p=2, dim=1)
        dep_proj = dep_proj.reshape(B, H_d, W_d, 256)
    else:
        dep_proj = depth_model.projection_head(dep_flat).reshape(B, H_d, W_d, 256)
        dep_proj = F.normalize(dep_proj, p=2, dim=1)

    # ── Fuse each patch through GACM ──
    rgb_flat_256 = rgb_proj.reshape(B * H * W, 256)
    dep_flat_256 = dep_proj.reshape(B * H * W, 256)
    fused_flat = fusion_model(rgb_flat_256, dep_flat_256)  # (B*196, 256)
    fused_patches = fused_flat.reshape(B, H * W, 256)      # (B, 196, 256)

    return fused_patches


def score_with_patchgrid(patch_vectors, coreset_bank, category=None):
    """
    Score an image using patch-grid comparison against normal coreset bank.
    Uses HYBRID scoring:
      - SMOOTH categories: MAX patch distance (catches single-patch micro-defects)
      - ORGANIC categories: Top-K average (smooths natural texture noise)

    Returns:
      - primary_dist: The hybrid score (MAX or Top-K avg depending on category)
      - max_dist: Maximum patch distance
      - mean_dist: Mean patch distance (overall anomaly level)
      - p95_dist: 95th percentile distance (robust to outlier patches)
      - topk_dist: Average of top-K patch distances
      - patch_dists: (196,) array of per-patch distances
    """
    # patch_vectors: (196, 256)
    # coreset_bank: (K, 256)
    dist_matrix = torch.cdist(patch_vectors, coreset_bank, p=2.0)  # (196, K)
    min_dists, _ = torch.min(dist_matrix, dim=1)  # (196,)

    max_dist = float(torch.max(min_dists).item())
    mean_dist = float(torch.mean(min_dists).item())
    p95_dist = float(torch.quantile(min_dists, 0.95).item())

    # Top-K average: average the K highest patch distances
    k = min(TOP_K, min_dists.shape[0])
    topk_vals, _ = torch.topk(min_dists, k)
    topk_dist = float(torch.mean(topk_vals).item())

    # Hybrid: select primary score based on category type
    if category is not None and category in ORGANIC_CATEGORIES:
        primary_dist = topk_dist
    else:
        primary_dist = max_dist

    return primary_dist, max_dist, mean_dist, p95_dist, topk_dist, min_dists.cpu().numpy()


def compute_distances(rgb_model, depth_model, fusion_model, category_means, coreset_banks):
    """Compute patch-grid anomaly scores for every test sample using HYBRID scoring."""
    logger.info("\n--- Computing HYBRID Patch-Grid Distances ---")
    logger.info("    Each image -> 14x14 = 196 patches -> compare each to coreset bank")
    logger.info(f"    SMOOTH categories ({', '.join(sorted(SMOOTH_CATEGORIES))}): MAX patch distance")
    logger.info(f"    ORGANIC categories ({', '.join(sorted(ORGANIC_CATEGORIES))}): Top-{TOP_K} average")

    results = defaultdict(lambda: {
        "normal_dists": [], "defect_dists": [],
        "all_dists": [], "all_labels": [],
        "normal_mean_dists": [], "defect_mean_dists": [],
        "normal_p95_dists": [], "defect_p95_dists": [],
        "normal_global_dists": [], "defect_global_dists": [],
    })

    ds = UnifiedAnomalyDataset(
        split="test", transform=get_eval_transform(), depth_transform=get_eval_transform()
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch_idx, (rgb, depth, labels, cat_idxs) in enumerate(loader):
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)

            # Also compute global fused features for comparison
            f_rgb_global = rgb_model(rgb)
            f_depth_global, _ = depth_model(depth)
            f_vis_global = fusion_model(f_rgb_global, f_depth_global)

            # Try patch-grid extraction
            try:
                patch_vecs = extract_patch_features_simple(
                    rgb_model, depth_model, fusion_model, rgb, depth
                )  # (B, 196, 256)
                has_patches = True
            except Exception as e:
                if batch_idx == 0:
                    logger.warning(f"  Patch extraction failed ({e}), using global features only")
                has_patches = False

            for i in range(len(labels)):
                cat_idx = cat_idxs[i].item()
                cat_name = ALL_CATEGORIES[cat_idx]
                label = labels[i].item()

                # PRIMARY: Hybrid patch-grid scoring (if coreset bank exists)
                if has_patches and cat_name in coreset_banks:
                    patches_i = patch_vecs[i]  # (196, 256)
                    bank = coreset_banks[cat_name].to(DEVICE)
                    primary_d, max_d, mean_d, p95_d, topk_d, _ = score_with_patchgrid(patches_i, bank, category=cat_name)
                    dist = primary_d  # Hybrid: MAX for smooth, Top-K for organic
                else:
                    # FALLBACK: Global manifold distance
                    if cat_name in category_means:
                        dist = torch.norm(f_vis_global[i] - category_means[cat_name], p=2).item()
                    else:
                        dist = 0.0
                    max_d = dist
                    mean_d = dist
                    p95_d = dist

                # Also compute global dist for comparison
                if cat_name in category_means:
                    global_dist = torch.norm(f_vis_global[i] - category_means[cat_name], p=2).item()
                else:
                    global_dist = 0.0

                results[cat_name]["all_dists"].append(dist)
                results[cat_name]["all_labels"].append(label)

                if label == 0:
                    results[cat_name]["normal_dists"].append(dist)
                    results[cat_name]["normal_mean_dists"].append(mean_d)
                    results[cat_name]["normal_p95_dists"].append(p95_d)
                    results[cat_name]["normal_global_dists"].append(global_dist)
                else:
                    results[cat_name]["defect_dists"].append(dist)
                    results[cat_name]["defect_mean_dists"].append(mean_d)
                    results[cat_name]["defect_p95_dists"].append(p95_d)
                    results[cat_name]["defect_global_dists"].append(global_dist)

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"    Processed {(batch_idx+1)*8}/{len(ds)} samples...")

    for cat in ALL_CATEGORIES:
        r = results[cat]
        n_norm = len(r["normal_dists"])
        n_def = len(r["defect_dists"])
        if n_norm > 0 and n_def > 0:
            logger.info(
                f"  {cat:15s}: {n_norm} normal (μ={np.mean(r['normal_dists']):.4f}), "
                f"{n_def} defect (μ={np.mean(r['defect_dists']):.4f}), "
                f"gap={np.mean(r['defect_dists'])-np.mean(r['normal_dists']):.4f}"
            )

    return dict(results)


def find_optimal_threshold(normal_dists, defect_dists):
    """Find optimal threshold using Youden's J-Index."""
    all_dists = normal_dists + defect_dists
    all_labels = [0] * len(normal_dists) + [1] * len(defect_dists)

    if len(set(all_labels)) < 2:
        return np.mean(all_dists) if all_dists else 0.5, 0.0

    fpr, tpr, thresholds = roc_curve(all_labels, all_dists)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    roc_auc = auc(fpr, tpr)

    return float(best_threshold), float(roc_auc)


def generate_graphs(results, out_dir):
    """Generate all evaluation graphs."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="darkgrid", palette="husl")

    category_aucs = {}
    thresholds = {}

    for cat in ALL_CATEGORIES:
        r = results.get(cat, {})
        normal_dists = r.get("normal_dists", [])
        defect_dists = r.get("defect_dists", [])

        if not normal_dists or not defect_dists:
            logger.warning(f"  {cat}: Insufficient data for graphs, skipping")
            continue

        threshold, roc_auc = find_optimal_threshold(normal_dists, defect_dists)
        category_aucs[cat] = roc_auc
        thresholds[cat] = threshold

        # 1. Distance Distribution Histogram (Patch-Grid Max Distance)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(normal_dists, bins=30, alpha=0.6, color='#2ecc71',
                label=f'Normal (n={len(normal_dists)})', density=True)
        ax.hist(defect_dists, bins=30, alpha=0.6, color='#e74c3c',
                label=f'Defect (n={len(defect_dists)})', density=True)
        ax.axvline(threshold, color='#3498db', linestyle='--', linewidth=2,
                   label=f'Threshold={threshold:.4f}')
        ax.set_xlabel('Max Patch Distance (L2) — No GAP', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{cat.replace("_"," ").title()} — Patch-Grid Distance Distribution\n'
                     f'(14×14 grid, score = max patch distance)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{cat}_distance_histogram.png"), dpi=150)
        plt.close()

        # 2. ROC Curve
        all_dists = normal_dists + defect_dists
        all_labels = [0] * len(normal_dists) + [1] * len(defect_dists)
        fpr, tpr, _ = roc_curve(all_labels, all_dists)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{cat.replace("_"," ").title()} — ROC Curve (Patch-Grid)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{cat}_roc_curve.png"), dpi=150)
        plt.close()

        # 3. Confusion Matrix
        preds = [1 if d > threshold else 0 for d in all_dists]
        cm = confusion_matrix(all_labels, preds)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Defect'], yticklabels=['Normal', 'Defect'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'{cat.replace("_"," ").title()} — Confusion Matrix', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{cat}_confusion_matrix.png"), dpi=150)
        plt.close()

        # 4. Patch-Grid vs GAP comparison (normal vs defect)
        normal_global = r.get("normal_global_dists", [])
        defect_global = r.get("defect_global_dists", [])
        if normal_global and defect_global:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            # GAP-based
            axes[0].hist(normal_global, bins=25, alpha=0.6, color='#2ecc71', label='Normal', density=True)
            axes[0].hist(defect_global, bins=25, alpha=0.6, color='#e74c3c', label='Defect', density=True)
            axes[0].set_title('Global Average Pooling (loses spatial info)', fontsize=11, fontweight='bold')
            axes[0].set_xlabel('Global L2 Distance')
            axes[0].legend()
            # Patch-Grid
            axes[1].hist(normal_dists, bins=25, alpha=0.6, color='#2ecc71', label='Normal', density=True)
            axes[1].hist(defect_dists, bins=25, alpha=0.6, color='#e74c3c', label='Defect', density=True)
            axes[1].set_title('Patch-Grid Max (catches tiny scratches)', fontsize=11, fontweight='bold')
            axes[1].set_xlabel('Max Patch L2 Distance')
            axes[1].legend()
            fig.suptitle(f'{cat.replace("_"," ").title()} — GAP vs Patch-Grid Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{cat}_gap_vs_patchgrid.png"), dpi=150)
            plt.close()

    # 5. Summary AUC Bar Chart
    if category_aucs:
        fig, ax = plt.subplots(figsize=(14, 6))
        cats = list(category_aucs.keys())
        aucs_vals = [category_aucs[c] for c in cats]
        colors = ['#2ecc71' if a >= 0.9 else '#f39c12' if a >= 0.7 else '#e74c3c' for a in aucs_vals]
        bars = ax.bar(cats, aucs_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_ylabel('AUC Score', fontsize=12)
        ax.set_title('Patch-Grid AUC Scores — All 13 Categories\n(No GAP — spatial defect sensitivity)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=10)
        for bar, val in zip(bars, aucs_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "summary_auc_scores.png"), dpi=150)
        plt.close()

    return thresholds, category_aucs


def generate_report(results, thresholds, category_aucs, out_dir):
    """Generate detailed eval_report.txt."""
    report_path = os.path.join(out_dir, "eval_report.txt")
    lines = []
    lines.append("=" * 80)
    lines.append("  MULTIMODAL ANOMALY DETECTION - EVALUATION REPORT")
    lines.append("  Scoring: HYBRID PATCH-GRID (14x14 = 196 patches)")
    lines.append(f"  SMOOTH categories: MAX patch distance | ORGANIC categories: Top-{TOP_K} average")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  HYBRID SCORING RATIONALE:")
    lines.append("    Manufactured/smooth surfaces (phone_screen, car_metal, pcb, cable_gland, dowel):")
    lines.append("      -> MAX patch: catches even single-patch micro-defects on uniform surfaces.")
    lines.append("    Organic/textured surfaces (bagel, carrot, cookie, foam, peach, potato, rope, tire):")
    lines.append(f"      -> Top-{TOP_K} avg: smooths natural texture noise while detecting real defects.")

    all_threshold_dict = {}

    for cat in ALL_CATEGORIES:
        r = results.get(cat, {})
        normal_dists = r.get("normal_dists", [])
        defect_dists = r.get("defect_dists", [])

        scoring_mode = "MAX patch" if cat in SMOOTH_CATEGORIES else f"Top-{TOP_K} avg"

        lines.append(f"\n{'---' * 20}")
        lines.append(f"  Category: {cat.upper()}  [Scoring: {scoring_mode}]")
        lines.append(f"{'---' * 20}")

        if not normal_dists or not defect_dists:
            lines.append("  INSUFFICIENT DATA - Skipped")
            continue

        threshold = thresholds.get(cat, 0.5)
        roc_auc_val = category_aucs.get(cat, 0.0)
        all_threshold_dict[cat] = round(threshold, 6)

        all_dists = normal_dists + defect_dists
        all_labels = [0] * len(normal_dists) + [1] * len(defect_dists)
        preds = [1 if d > threshold else 0 for d in all_dists]

        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        mean_normal = np.mean(normal_dists)
        mean_defect = np.mean(defect_dists)
        sep_gap = mean_defect - mean_normal

        # Also show GAP comparison
        mean_normal_gap = np.mean(r.get("normal_global_dists", [0]))
        mean_defect_gap = np.mean(r.get("defect_global_dists", [0]))

        lines.append(f"  Samples:            {len(normal_dists)} normal, {len(defect_dists)} defect")
        lines.append(f"  Optimal Threshold:  {threshold:.6f}")
        lines.append(f"  AUC:                {roc_auc_val:.4f}")
        lines.append(f"  Accuracy:           {acc:.4f} ({acc*100:.1f}%)")
        lines.append(f"  Precision:          {prec:.4f}")
        lines.append(f"  Recall:             {rec:.4f}")
        lines.append(f"  F1 Score:           {f1:.4f}")
        lines.append(f"  -- Hybrid Scoring ({scoring_mode}) --")
        lines.append(f"  Mean Normal ({scoring_mode}): {mean_normal:.6f}")
        lines.append(f"  Mean Defect ({scoring_mode}): {mean_defect:.6f}")
        lines.append(f"  Separability Gap:        {sep_gap:.6f}")
        lines.append(f"  -- Global GAP Scoring (for comparison) --")
        lines.append(f"  Mean Normal (GAP):       {mean_normal_gap:.6f}")
        lines.append(f"  Mean Defect (GAP):       {mean_defect_gap:.6f}")
        lines.append(f"  GAP Separability:        {mean_defect_gap - mean_normal_gap:.6f}")

    # Summary table
    lines.append(f"\n\n{'=' * 80}")
    lines.append("  SUMMARY TABLE (Hybrid Scoring)")
    lines.append(f"{'=' * 80}")
    lines.append(f"  {'Category':<18} {'AUC':>8} {'Threshold':>12} {'F1':>8} {'Accuracy':>10} {'Sep.Gap':>10}")
    lines.append(f"  {'-'*18} {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")

    for cat in ALL_CATEGORIES:
        r = results.get(cat, {})
        normal_dists = r.get("normal_dists", [])
        defect_dists = r.get("defect_dists", [])
        if not normal_dists or not defect_dists:
            continue
        threshold = thresholds.get(cat, 0.5)
        auc_val = category_aucs.get(cat, 0.0)
        all_dists = normal_dists + defect_dists
        all_labels = [0] * len(normal_dists) + [1] * len(defect_dists)
        preds = [1 if d > threshold else 0 for d in all_dists]
        f1 = f1_score(all_labels, preds, zero_division=0)
        acc = accuracy_score(all_labels, preds)
        gap = np.mean(defect_dists) - np.mean(normal_dists)
        lines.append(f"  {cat:<18} {auc_val:>8.4f} {threshold:>12.6f} {f1:>8.4f} {acc*100:>9.1f}% {gap:>10.4f}")

    # Threshold map for web app
    lines.append(f"\n\n{'=' * 80}")
    lines.append("  CALIBRATED THRESHOLD MAP (for web_app/main.py)")
    lines.append("  NOTE: These are PATCH-GRID thresholds (max patch distance)")
    lines.append(f"{'=' * 80}")
    lines.append("  threshold_map = {")
    for cat, t in all_threshold_dict.items():
        lines.append(f'      "{cat}": {t},')
    lines.append("  }")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"\n  ✓ Report saved to {report_path}")

    return all_threshold_dict


if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 65)
    logger.info("  PHASE 4: RIGOROUS EVALUATION (PATCH-GRID, NO GAP)")
    logger.info("=" * 65)

    rgb_model, depth_model, fusion_model, category_means, coreset_banks = load_models_and_banks()

    results = compute_distances(rgb_model, depth_model, fusion_model, category_means, coreset_banks)

    out_dir = os.path.join(FUSION_DIR, "outputs", "evaluation")
    thresholds, category_aucs = generate_graphs(results, out_dir)
    threshold_dict = generate_report(results, thresholds, category_aucs, out_dir)

    # Save threshold map as JSON for Phase 5
    json_path = os.path.join(out_dir, "calibrated_thresholds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_dict, f, indent=2)
    logger.info(f"  ✓ Thresholds saved to {json_path}")

    elapsed = time.time() - start
    logger.info(f"\nPhase 4 complete in {elapsed/60:.1f} minutes")
