"""
compute_means_and_banks.py — Phase 3: Compute Manifold Means & PatchCore Banks.

For each of the 13 categories:
  1. Computes the L2-normalized mean fused feature vector from normal training images
  2. Builds PatchCore coreset memory banks for spatial anomaly detection
"""

import os
import sys
import json
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ComputeMeans")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SDC_DIR = os.path.join(BASE_DIR, "sdc_project")
DEPTH_DIR = os.path.join(BASE_DIR, "depth_encoder_share", "depth_encoder")
FUSION_DIR = os.path.join(BASE_DIR, "multimodal_fusion_pipeline")

for d in [SDC_DIR, os.path.join(DEPTH_DIR, "models"), FUSION_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from unified_dataset import (
    UnifiedAnomalyDataset, ALL_CATEGORIES, get_eval_transform
)


def load_trained_models():
    """Load all trained models from checkpoints."""
    # RGB
    # pyrefly: ignore [missing-import]
    from feature_head import RGBFeatureExtractor
    rgb_model = RGBFeatureExtractor(encoder_pretrained=True, freeze_up_to=8, feature_dim=256).to(DEVICE)
    rgb_ckpt = os.path.join(SDC_DIR, "outputs", "checkpoints", "best_model.pth")
    if os.path.isfile(rgb_ckpt):
        ckpt = torch.load(rgb_ckpt, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        rgb_model.load_state_dict(sd)
        logger.info(f"  Loaded RGB encoder from {rgb_ckpt}")
    rgb_model.eval()

    # Depth
    # pyrefly: ignore [missing-import]
    from depth_encoder import DepthEncoder
    depth_model = DepthEncoder(embedding_dim=256, pretrained=True).to(DEVICE)
    depth_ckpt = os.path.join(DEPTH_DIR, "checkpoints", "best.pt")
    if os.path.isfile(depth_ckpt):
        ckpt = torch.load(depth_ckpt, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        # Strip 'encoder.' prefix if present
        cleaned = {}
        for k, v in sd.items():
            cleaned[k.replace("encoder.", "") if k.startswith("encoder.") else k] = v
        depth_model.load_state_dict(cleaned, strict=False)
        logger.info(f"  Loaded Depth encoder from {depth_ckpt}")
    depth_model.eval()

    # Fusion (GACM)
    # pyrefly: ignore [missing-import]
    from gacm import LightweightGACM
    import torch.nn as nn

    class VisualFusionPipeline(nn.Module):
        def __init__(self, dim=256, hidden_dim=512):
            super().__init__()
            self.gacm = LightweightGACM(dim=dim, hidden_dim=hidden_dim)
        def forward(self, f_rgb, f_depth):
            return self.gacm(f_rgb, f_depth)

    fusion_model = VisualFusionPipeline(dim=256, hidden_dim=512).to(DEVICE)
    fusion_ckpt = os.path.join(FUSION_DIR, "outputs", "checkpoints", "best_fusion_model_finetuned.pth")
    if not os.path.isfile(fusion_ckpt):
        fusion_ckpt = os.path.join(FUSION_DIR, "outputs", "checkpoints", "best_fusion_model.pth")
    if os.path.isfile(fusion_ckpt):
        ckpt = torch.load(fusion_ckpt, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        fusion_model.load_state_dict(sd)
        logger.info(f"  Loaded Fusion model from {fusion_ckpt}")
    fusion_model.eval()

    return rgb_model, depth_model, fusion_model


def compute_category_means(rgb_model, depth_model, fusion_model):
    """Compute L2-normalized mean fused feature vector for each category's normal images."""
    logger.info("\n--- Computing Per-Category Manifold Means ---")

    means_dict = {}
    means_np = {}

    for cat in ALL_CATEGORIES:
        ds = UnifiedAnomalyDataset(
            split="train", categories=[cat], normal_only=True,
            transform=get_eval_transform(), depth_transform=get_eval_transform()
        )

        if len(ds) == 0:
            logger.warning(f"  {cat}: No normal training samples found, skipping")
            continue

        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        all_feats = []

        with torch.no_grad():
            for rgb, depth, labels, cat_idxs in loader:
                rgb = rgb.to(DEVICE)
                depth = depth.to(DEVICE)
                f_rgb = rgb_model(rgb)
                f_depth, _ = depth_model(depth)
                f_vis = fusion_model(f_rgb, f_depth)
                all_feats.append(f_vis.cpu())

        all_feats = torch.cat(all_feats, dim=0)  # (N, 256)
        mean_vec = all_feats.mean(dim=0)
        mean_vec = F.normalize(mean_vec, p=2, dim=0)

        means_dict[cat] = mean_vec.tolist()
        means_np[cat] = mean_vec.numpy()

        logger.info(f"  {cat:15s}: {len(ds)} normal samples -> mean norm = {mean_vec.norm():.4f}")

    # Save JSON
    out_dir = os.path.join(FUSION_DIR, "outputs", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "all_category_means.json")
    with open(json_path, "w") as f:
        json.dump(means_dict, f, indent=2)
    logger.info(f"\n  ✓ Saved means to {json_path}")

    # Also save as numpy
    npy_path = os.path.join(out_dir, "all_category_means.npy")
    np.save(npy_path, means_np)

    return means_dict

def get_augmentation_transforms(n_augments=5):
    """Generate multiple augmentation transforms for enriching coreset banks."""
    from torchvision import transforms as T
    augs = []
    for _ in range(n_augments):
        augs.append(T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(30),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]))
    return augs


def build_patchcore_banks(rgb_model, depth_model, fusion_model):
    """Build PatchCore coreset memory banks for all 13 categories.
    
    For PCB, builds an enriched bank by generating 5 augmented views
    of each normal training image with a higher 15% coreset sampling ratio.
    """
    # Enrichment config — PCB and phone_screen
    ENRICH_CATEGORIES = {"pcb", "phone_screen"}
    ENRICH_N_AUGMENTS = 5
    ENRICH_SAMPLING_RATIO = 0.15

    logger.info("\n--- Building PatchCore Coreset Banks ---")
    logger.info(f"    Enriched categories ({', '.join(ENRICH_CATEGORIES)}): {ENRICH_N_AUGMENTS}x augmented views, {ENRICH_SAMPLING_RATIO*100:.0f}% coreset")

    sys.path.insert(0, FUSION_DIR)
    from patchcore_engine import MultimodalPatchExtractor, KCenterGreedyCoreset

    extractor = MultimodalPatchExtractor(rgb_model, depth_model, fusion_model, target_size=(14, 14))
    coreset_selector_default = KCenterGreedyCoreset(sampling_ratio=0.10)
    coreset_selector_enriched = KCenterGreedyCoreset(sampling_ratio=ENRICH_SAMPLING_RATIO)

    aug_transforms = get_augmentation_transforms(ENRICH_N_AUGMENTS)

    coreset_banks = {}

    for cat in ALL_CATEGORIES:
        needs_enrichment = cat in ENRICH_CATEGORIES

        # --- Standard pass (eval transform, no augmentation) ---
        ds = UnifiedAnomalyDataset(
            split="train", categories=[cat], normal_only=True,
            transform=get_eval_transform(), depth_transform=get_eval_transform()
        )

        if len(ds) == 0:
            logger.warning(f"  {cat}: No samples, skipping PatchCore bank")
            continue

        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        all_patches = []

        with torch.no_grad():
            for rgb, depth, labels, cat_idxs in loader:
                rgb = rgb.to(DEVICE)
                depth = depth.to(DEVICE)
                try:
                    patch_vecs = extractor.extract_patch_features(rgb, depth)  # (B, 196, 256)
                    all_patches.append(patch_vecs.cpu().reshape(-1, 256))
                except Exception as e:
                    logger.warning(f"  {cat}: PatchCore extraction error: {e}")
                    continue

        base_count = sum(p.shape[0] for p in all_patches) if all_patches else 0

        # --- Augmented passes (only for enriched categories) ---
        if needs_enrichment and len(ds) > 0:
            logger.info(f"  {cat:15s}: Enriching with {ENRICH_N_AUGMENTS} augmented views...")
            for aug_idx, aug_tf in enumerate(aug_transforms):
                ds_aug = UnifiedAnomalyDataset(
                    split="train", categories=[cat], normal_only=True,
                    transform=aug_tf, depth_transform=aug_tf
                )
                loader_aug = DataLoader(ds_aug, batch_size=8, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for rgb, depth, labels, cat_idxs in loader_aug:
                        rgb = rgb.to(DEVICE)
                        depth = depth.to(DEVICE)
                        try:
                            patch_vecs = extractor.extract_patch_features(rgb, depth)
                            all_patches.append(patch_vecs.cpu().reshape(-1, 256))
                        except Exception:
                            continue

        if not all_patches:
            continue

        all_patches = torch.cat(all_patches, dim=0)  # (N*196, 256)
        selector = coreset_selector_enriched if needs_enrichment else coreset_selector_default

        if needs_enrichment:
            logger.info(f"  {cat:15s}: {base_count} base + {all_patches.shape[0] - base_count} augmented = {all_patches.shape[0]} total patches. Extracting coreset...")
        else:
            logger.info(f"  {cat:15s}: {all_patches.shape[0]} total patches. Extracting coreset...")

        coreset = selector.sample(all_patches)
        coreset_banks[cat] = coreset
        logger.info(f"  {cat:15s}: Reduced to {coreset.shape[0]} coreset patches")

    # Save
    out_path = os.path.join(FUSION_DIR, "outputs", "checkpoints", "patchcore_coreset_banks.pt")
    torch.save(coreset_banks, out_path)
    logger.info(f"\n  ✓ Saved PatchCore banks to {out_path} ({len(coreset_banks)} categories)")

    return coreset_banks


if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 65)
    logger.info("  PHASE 3: MANIFOLD MEANS & PATCHCORE BANKS")
    logger.info("=" * 65)

    rgb_model, depth_model, fusion_model = load_trained_models()
    compute_category_means(rgb_model, depth_model, fusion_model)
    build_patchcore_banks(rgb_model, depth_model, fusion_model)

    elapsed = time.time() - start
    logger.info(f"\nPhase 3 complete in {elapsed/60:.1f} minutes")

