"""
train_pipeline.py — Master Training Script for All 4 Pipeline Components.

Sequentially trains:
  Step A: RGB Encoder (MobileNetV3 + FeatureHead) on all 13 categories
  Step B: Depth Encoder on all 13 categories (real depth / grayscale proxy)
  Step C: CLIP projection + OCTA text adapter alignment
  Step D: GACM Visual Fusion Model on all 13 categories

Uses RTX GPU with early stopping and validation-based best model saving.
"""

import os
import sys
import time
import json
import logging
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainPipeline")

# ── Setup paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SDC_DIR = os.path.join(BASE_DIR, "sdc_project")
DEPTH_DIR = os.path.join(BASE_DIR, "depth_encoder_share", "depth_encoder")
FUSION_DIR = os.path.join(BASE_DIR, "multimodal_fusion_pipeline")
TEXT_DIR = os.path.join(BASE_DIR, "member1_text_pipeline", "models")

for d in [SDC_DIR, DEPTH_DIR, FUSION_DIR, TEXT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Import project modules
from unified_dataset import (
    UnifiedAnomalyDataset, create_dataloaders,
    ALL_CATEGORIES, CATEGORY_TO_IDX, get_eval_transform
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")


# ══════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

class ContrastiveAnomalyLoss(nn.Module):
    """
    Loss for encoder training:
    - Normal samples: cosine consistency + compactness around running EMA center
    - Defect samples: push away from center (margin-based)
    """
    def __init__(self, feature_dim=256, num_categories=13, margin=0.5):
        super().__init__()
        self.margin = margin
        # Per-category running centers
        self.register_buffer(
            "centers", torch.zeros(num_categories, feature_dim)
        )
        self.register_buffer(
            "center_counts", torch.zeros(num_categories)
        )
        self.ema_decay = 0.99

    def forward(self, features, labels, cat_idxs):
        """
        features: (B, 256) L2-normalized
        labels:   (B,) 0=normal, 1=defect
        cat_idxs: (B,) category index
        """
        loss = torch.tensor(0.0, device=features.device)
        count = 0

        for cat_id in cat_idxs.unique():
            mask_cat = (cat_idxs == cat_id)
            cat_feats = features[mask_cat]
            cat_labels = labels[mask_cat]

            normal_mask = (cat_labels == 0)
            defect_mask = (cat_labels == 1)

            normal_feats = cat_feats[normal_mask]
            defect_feats = cat_feats[defect_mask]

            center = self.centers[cat_id].unsqueeze(0)

            # Normal: compactness — pull toward center
            if len(normal_feats) > 0:
                compact_loss = ((normal_feats - center) ** 2).sum(dim=1).mean()
                loss = loss + compact_loss
                count += 1

                # Update center via EMA
                with torch.no_grad():
                    batch_mean = normal_feats.mean(dim=0)
                    if self.center_counts[cat_id] == 0:
                        self.centers[cat_id] = batch_mean
                    else:
                        self.centers[cat_id] = (
                            self.ema_decay * self.centers[cat_id] +
                            (1 - self.ema_decay) * batch_mean
                        )
                    self.center_counts[cat_id] += 1

            # Defect: push away from center (margin-based)
            if len(defect_feats) > 0 and self.center_counts[cat_id] > 0:
                dist = torch.norm(defect_feats - center, dim=1)
                push_loss = F.relu(self.margin - dist).mean()
                loss = loss + 0.5 * push_loss
                count += 1

            # Intra-normal consistency (cosine)
            if len(normal_feats) > 1:
                # Random pairs
                n = len(normal_feats)
                idx1 = torch.randperm(n, device=features.device)[:min(n, 8)]
                idx2 = torch.randperm(n, device=features.device)[:min(n, 8)]
                min_len = min(len(idx1), len(idx2))
                cos_loss = (1.0 - F.cosine_similarity(
                    normal_feats[idx1[:min_len]], normal_feats[idx2[:min_len]], dim=1
                )).mean()
                loss = loss + cos_loss
                count += 1

        return loss / max(count, 1)


# ══════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # not improved


# ══════════════════════════════════════════════════════════════════════
# STEP A: TRAIN RGB ENCODER
# ══════════════════════════════════════════════════════════════════════

def train_rgb_encoder(max_epochs=50, batch_size=16, lr=3e-4):
    logger.info("=" * 65)
    logger.info("  STEP A: TRAINING RGB ENCODER ON ALL 13 CATEGORIES")
    logger.info("=" * 65)

    # Import RGB model
    sys.path.insert(0, SDC_DIR)
    from feature_head import RGBFeatureExtractor

    model = RGBFeatureExtractor(
        encoder_pretrained=True,
        freeze_up_to=8,
        feature_dim=256
    ).to(DEVICE)
    model.train()

    # Dataloaders
    train_loader, _ = create_dataloaders("train", batch_size=batch_size, balanced=True)
    val_loader, _ = create_dataloaders("val", batch_size=batch_size * 2, balanced=False)

    logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer with differential LR
    param_groups = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": lr * 0.1},
        {"params": model.head.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    loss_fn = ContrastiveAnomalyLoss(feature_dim=256).to(DEVICE)
    early_stop = EarlyStopping(patience=15)

    ckpt_dir = os.path.join(SDC_DIR, "outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for rgb, depth, labels, cat_idxs in train_loader:
            rgb = rgb.to(DEVICE)
            labels = labels.to(DEVICE)
            cat_idxs = cat_idxs.to(DEVICE)

            features = model(rgb)  # (B, 256)
            loss = loss_fn(features, labels, cat_idxs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for rgb, depth, labels, cat_idxs in val_loader:
                rgb = rgb.to(DEVICE)
                labels = labels.to(DEVICE)
                cat_idxs = cat_idxs.to(DEVICE)
                features = model(rgb)
                loss = loss_fn(features, labels, cat_idxs)
                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)

        improved = early_stop.step(avg_val)
        marker = " ★" if improved else ""

        if epoch <= 5 or epoch % 5 == 0 or improved or early_stop.should_stop:
            logger.info(
                f"  Epoch {epoch:3d}/{max_epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}{marker}"
            )

        if improved:
            best_model_state = copy.deepcopy(model.state_dict())

        if early_stop.should_stop:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Save best model
    if best_model_state is not None:
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
        torch.save({"model_state_dict": best_model_state, "epoch": epoch}, ckpt_path)
        model.load_state_dict(best_model_state)
        logger.info(f"  ✓ RGB Encoder saved to {ckpt_path}")

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════
# STEP B: TRAIN DEPTH ENCODER
# ══════════════════════════════════════════════════════════════════════

def train_depth_encoder(max_epochs=50, batch_size=16, lr=3e-4):
    logger.info("=" * 65)
    logger.info("  STEP B: TRAINING DEPTH ENCODER ON ALL 13 CATEGORIES")
    logger.info("=" * 65)

    sys.path.insert(0, os.path.join(DEPTH_DIR, "models"))
    from depth_encoder import DepthEncoder

    model = DepthEncoder(embedding_dim=256, pretrained=True).to(DEVICE)
    model.train()

    train_loader, _ = create_dataloaders("train", batch_size=batch_size, balanced=True)
    val_loader, _ = create_dataloaders("val", batch_size=batch_size * 2, balanced=False)

    logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    loss_fn = ContrastiveAnomalyLoss(feature_dim=256).to(DEVICE)
    early_stop = EarlyStopping(patience=15)

    ckpt_dir = os.path.join(DEPTH_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for rgb, depth, labels, cat_idxs in train_loader:
            depth = depth.to(DEVICE)
            labels = labels.to(DEVICE)
            cat_idxs = cat_idxs.to(DEVICE)

            embedding, _ = model(depth)  # (B, 256)
            loss = loss_fn(embedding, labels, cat_idxs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for rgb, depth, labels, cat_idxs in val_loader:
                depth = depth.to(DEVICE)
                labels = labels.to(DEVICE)
                cat_idxs = cat_idxs.to(DEVICE)
                embedding, _ = model(depth)
                loss = loss_fn(embedding, labels, cat_idxs)
                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        improved = early_stop.step(avg_val)
        marker = " ★" if improved else ""

        if epoch <= 5 or epoch % 5 == 0 or improved or early_stop.should_stop:
            logger.info(
                f"  Epoch {epoch:3d}/{max_epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}{marker}"
            )

        if improved:
            best_model_state = copy.deepcopy(model.state_dict())

        if early_stop.should_stop:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        ckpt_path = os.path.join(ckpt_dir, "best.pt")
        torch.save({"model_state_dict": best_model_state, "epoch": epoch}, ckpt_path)
        model.load_state_dict(best_model_state)
        logger.info(f"  ✓ Depth Encoder saved to {ckpt_path}")

    model.eval()
    model.remove_hooks()
    return model


# ══════════════════════════════════════════════════════════════════════
# STEP C: TRAIN CLIP PROJECTION + OCTA TEXT PIPELINE
# ══════════════════════════════════════════════════════════════════════

def train_text_pipeline(rgb_model, max_epochs=30, lr=1e-3):
    logger.info("=" * 65)
    logger.info("  STEP C: TRAINING CLIP PROJECTION + OCTA TEXT PIPELINE")
    logger.info("=" * 65)

    sys.path.insert(0, TEXT_DIR)
    from clip_encoder import CLIPTextEncoder
    from octa import OCTA
    from prompt_templates import PromptGenerator

    # CLIP encoder with trainable projection
    clip_encoder = CLIPTextEncoder(output_dim=256, device=DEVICE)
    octa = OCTA(dim=256).to(DEVICE)
    prompt_gen = PromptGenerator(classes=ALL_CATEGORIES)

    # Trainable params: CLIP projection + OCTA
    trainable_params = list(clip_encoder.projection.parameters()) + list(octa.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    # Use the frozen RGB model to get visual features for alignment
    rgb_model.eval()

    train_loader, _ = create_dataloaders("train", batch_size=16, balanced=True, normal_only=False)
    val_loader, _ = create_dataloaders("val", batch_size=32, balanced=False, normal_only=False)

    early_stop = EarlyStopping(patience=10)
    best_state = None
    ckpt_dir = os.path.join(TEXT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_octa.pth")
    
    start_epoch = 1
    if os.path.exists(ckpt_path):
        logger.info(f"  Found existing text pipeline checkpoint at {ckpt_path}. Resuming!")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if "clip_projection" in checkpoint:
            clip_encoder.projection.load_state_dict(checkpoint["clip_projection"])
        if "octa" in checkpoint:
            octa.load_state_dict(checkpoint["octa"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_val_loss" in checkpoint:
            early_stop.best_loss = checkpoint["best_val_loss"]

    for epoch in range(start_epoch, max_epochs + 1):
        octa.train()
        clip_encoder.projection.train() if hasattr(clip_encoder.projection, 'train') else None
        train_loss_total = 0.0
        n_batches = 0

        for rgb, depth, labels, cat_idxs in train_loader:
            rgb = rgb.to(DEVICE)
            labels = labels.to(DEVICE)
            cat_idxs = cat_idxs.to(DEVICE)

            # Get visual features from frozen RGB encoder
            with torch.no_grad():
                f_rgb = rgb_model(rgb)  # (B, 256)

            loss = torch.tensor(0.0, device=DEVICE)
            batch_count = 0

            for cat_id in cat_idxs.unique():
                cat_name = ALL_CATEGORIES[cat_id.item()]
                mask = (cat_idxs == cat_id)
                cat_feats = f_rgb[mask]
                cat_labels = labels[mask]

                # Get normal text embedding via OCTA
                normal_prompts = prompt_gen.generate_prompts(cat_name)[:21]  # Limit for speed
                text_embeds = clip_encoder(normal_prompts)  # (N, 256)
                f_text = octa(text_embeds)  # (1, 256)
                f_text_norm = F.normalize(f_text, p=2, dim=1)

                # Normal visual features should align with normal text
                normal_feats = cat_feats[cat_labels == 0]
                defect_feats = cat_feats[cat_labels == 1]

                if len(normal_feats) > 0:
                    sim_normal = F.cosine_similarity(
                        normal_feats, f_text_norm.expand_as(normal_feats), dim=1
                    )
                    align_loss = (1.0 - sim_normal).mean()
                    loss = loss + align_loss
                    batch_count += 1

                # Defect visual features should NOT align with normal text
                if len(defect_feats) > 0:
                    sim_defect = F.cosine_similarity(
                        defect_feats, f_text_norm.expand_as(defect_feats), dim=1
                    )
                    push_loss = F.relu(sim_defect - 0.3).mean()
                    loss = loss + 0.5 * push_loss
                    batch_count += 1

            if batch_count > 0:
                loss = loss / batch_count
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            train_loss_total += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss_total / max(n_batches, 1)

        # Validation
        octa.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for rgb, depth, labels, cat_idxs in val_loader:
                rgb = rgb.to(DEVICE)
                labels = labels.to(DEVICE)
                cat_idxs = cat_idxs.to(DEVICE)
                f_rgb = rgb_model(rgb)

                batch_loss = torch.tensor(0.0, device=DEVICE)
                bc = 0
                for cat_id in cat_idxs.unique():
                    cat_name = ALL_CATEGORIES[cat_id.item()]
                    mask = (cat_idxs == cat_id)
                    cat_feats = f_rgb[mask]
                    cat_labels = labels[mask]

                    normal_prompts = prompt_gen.generate_prompts(cat_name)[:21]
                    text_embeds = clip_encoder(normal_prompts)
                    f_text = octa(text_embeds)
                    f_text_norm = F.normalize(f_text, p=2, dim=1)

                    normal_feats = cat_feats[cat_labels == 0]
                    if len(normal_feats) > 0:
                        sim = F.cosine_similarity(
                            normal_feats, f_text_norm.expand_as(normal_feats), dim=1
                        )
                        batch_loss = batch_loss + (1.0 - sim).mean()
                        bc += 1

                if bc > 0:
                    val_loss_total += (batch_loss / bc).item()
                    val_batches += 1

        avg_val = val_loss_total / max(val_batches, 1)
        improved = early_stop.step(avg_val)
        marker = " ★" if improved else ""

        if epoch <= 3 or epoch % 5 == 0 or improved or early_stop.should_stop:
            logger.info(
                f"  Epoch {epoch:3d}/{max_epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}{marker}"
            )

        if improved:
            best_state = {
                "octa_state_dict": copy.deepcopy(octa.state_dict()),
                "clip_projection_state_dict": copy.deepcopy(clip_encoder.projection.state_dict()),
                "epoch": epoch
            }

        if early_stop.should_stop:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        ckpt_path = os.path.join(ckpt_dir, "best_octa.pth")
        torch.save(best_state, ckpt_path)
        octa.load_state_dict(best_state["octa_state_dict"])
        clip_encoder.projection.load_state_dict(best_state["clip_projection_state_dict"])
        logger.info(f"  ✓ OCTA + CLIP projection saved to {ckpt_path}")

    octa.eval()
    return clip_encoder, octa


# ══════════════════════════════════════════════════════════════════════
# STEP D: TRAIN GACM FUSION MODEL
# ══════════════════════════════════════════════════════════════════════

def train_gacm_fusion(rgb_model, depth_model, max_epochs=50, batch_size=16, lr=1e-4):
    logger.info("=" * 65)
    logger.info("  STEP D: TRAINING GACM FUSION MODEL ON ALL 13 CATEGORIES")
    logger.info("=" * 65)

    sys.path.insert(0, FUSION_DIR)
    from gacm import LightweightGACM

    class VisualFusionPipeline(nn.Module):
        def __init__(self, dim=256, hidden_dim=512):
            super().__init__()
            self.gacm = LightweightGACM(dim=dim, hidden_dim=hidden_dim)
        def forward(self, f_rgb, f_depth):
            return self.gacm(f_rgb, f_depth)

    fusion_model = VisualFusionPipeline(dim=256, hidden_dim=512).to(DEVICE)
    fusion_model.train()

    rgb_model.eval()
    depth_model.eval()

    # Prepare dataloaders — normal-only for fusion consistency training
    train_loader, _ = create_dataloaders("train", batch_size=batch_size, balanced=True, normal_only=True)
    val_loader, _ = create_dataloaders("val", batch_size=batch_size * 2, balanced=False, normal_only=True)

    logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # EMA center for compactness loss (per category)
    centers = torch.zeros(13, 256, device=DEVICE)
    center_counts = torch.zeros(13, device=DEVICE)
    ema_decay = 0.99

    early_stop = EarlyStopping(patience=15)
    best_model_state = None

    ckpt_dir = os.path.join(FUSION_DIR, "outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        fusion_model.train()
        train_loss = 0.0
        n_batches = 0

        for rgb, depth, labels, cat_idxs in train_loader:
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)
            cat_idxs = cat_idxs.to(DEVICE)

            with torch.no_grad():
                f_rgb = rgb_model(rgb)
                f_depth, _ = depth_model(depth)

            f_vis = fusion_model(f_rgb, f_depth)

            # Consistency loss
            cos_rgb = (1.0 - F.cosine_similarity(f_vis, f_rgb, dim=1)).mean()
            cos_depth = (1.0 - F.cosine_similarity(f_vis, f_depth, dim=1)).mean()
            consistency_loss = (cos_rgb + cos_depth) / 2.0

            # Per-category compactness
            compact_loss = torch.tensor(0.0, device=DEVICE)
            n_cats = 0
            for cat_id in cat_idxs.unique():
                mask = (cat_idxs == cat_id)
                cat_feats = f_vis[mask]
                center = centers[cat_id].unsqueeze(0)
                compact_loss = compact_loss + ((cat_feats - center) ** 2).sum(dim=1).mean()
                n_cats += 1

                with torch.no_grad():
                    batch_mean = cat_feats.mean(dim=0)
                    if center_counts[cat_id] == 0:
                        centers[cat_id] = batch_mean
                    else:
                        centers[cat_id] = ema_decay * centers[cat_id] + (1 - ema_decay) * batch_mean
                    center_counts[cat_id] += 1

            if n_cats > 0:
                compact_loss = compact_loss / n_cats

            loss = consistency_loss + 0.5 * compact_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        fusion_model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for rgb, depth, labels, cat_idxs in val_loader:
                rgb = rgb.to(DEVICE)
                depth = depth.to(DEVICE)
                cat_idxs = cat_idxs.to(DEVICE)

                f_rgb = rgb_model(rgb)
                f_depth, _ = depth_model(depth)
                f_vis = fusion_model(f_rgb, f_depth)

                cos_rgb = (1.0 - F.cosine_similarity(f_vis, f_rgb, dim=1)).mean()
                cos_depth = (1.0 - F.cosine_similarity(f_vis, f_depth, dim=1)).mean()
                consistency = (cos_rgb + cos_depth) / 2.0

                comp = torch.tensor(0.0, device=DEVICE)
                nc = 0
                for cat_id in cat_idxs.unique():
                    mask = (cat_idxs == cat_id)
                    cat_feats = f_vis[mask]
                    center = centers[cat_id].unsqueeze(0)
                    comp = comp + ((cat_feats - center) ** 2).sum(dim=1).mean()
                    nc += 1
                if nc > 0:
                    comp = comp / nc

                val_loss += (consistency + 0.5 * comp).item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        improved = early_stop.step(avg_val)
        marker = " ★" if improved else ""

        if epoch <= 5 or epoch % 5 == 0 or improved or early_stop.should_stop:
            logger.info(
                f"  Epoch {epoch:3d}/{max_epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}{marker}"
            )

        if improved:
            best_model_state = copy.deepcopy(fusion_model.state_dict())

        if early_stop.should_stop:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        ckpt_path = os.path.join(ckpt_dir, "best_fusion_model_finetuned.pth")
        torch.save({
            "model_state_dict": best_model_state,
            "epoch": epoch,
        }, ckpt_path)
        fusion_model.load_state_dict(best_model_state)
        logger.info(f"  ✓ GACM Fusion saved to {ckpt_path}")

    fusion_model.eval()
    return fusion_model


# ══════════════════════════════════════════════════════════════════════
# MAIN — ORCHESTRATE ALL STEPS
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 65)
    logger.info("  MULTIMODAL PIPELINE TRAINING (PARALLEL RGB/DEPTH)")
    logger.info("=" * 65)

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # ── PARALLEL STEP A & B (ALREADY COMPLETED) ──
    logger.info("  Skipping RGB/Depth training (already completed). Loading trained models...")
    # p_rgb = mp.Process(target=train_rgb_encoder, args=(50, 16, 3e-4))
    # p_depth = mp.Process(target=train_depth_encoder, args=(50, 16, 3e-4))

    # p_rgb.start()
    # p_depth.start()

    # p_rgb.join()
    # p_depth.join()

    logger.info("  Parallel encoder training complete. Loading trained models...")
    
    # Load trained RGB
    sys.path.insert(0, SDC_DIR)
    from feature_head import RGBFeatureExtractor
    rgb_model = RGBFeatureExtractor(encoder_pretrained=True, freeze_up_to=8, feature_dim=256).to(DEVICE)
    rgb_ckpt = torch.load(os.path.join(SDC_DIR, "outputs", "checkpoints", "best_model.pth"), map_location=DEVICE, weights_only=False)
    rgb_model.load_state_dict(rgb_ckpt.get("model_state_dict", rgb_ckpt))
    rgb_model.eval()

    # Load trained Depth
    sys.path.insert(0, os.path.join(DEPTH_DIR, "models"))
    from depth_encoder import DepthEncoder
    depth_model = DepthEncoder(embedding_dim=256, pretrained=True).to(DEVICE)
    depth_ckpt = torch.load(os.path.join(DEPTH_DIR, "checkpoints", "best.pt"), map_location=DEVICE, weights_only=False)
    sd = depth_ckpt.get("model_state_dict", depth_ckpt)
    cleaned = {k.replace("encoder.", "") if k.startswith("encoder.") else k: v for k, v in sd.items()}
    depth_model.load_state_dict(cleaned, strict=False)
    depth_model.eval()

    # Re-register hooks for depth model (needed for GACM feature extraction)
    depth_model._register_feature_hooks()

    # ── SEQUENTIAL STEP C & D ──
    # Step C
    clip_encoder, octa_model = train_text_pipeline(rgb_model, max_epochs=30, lr=1e-3)
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Step D
    fusion_model = train_gacm_fusion(rgb_model, depth_model, max_epochs=50, batch_size=16, lr=1e-4)

    elapsed = time.time() - start
    logger.info(f"\n{'=' * 65}")
    logger.info(f"  ALL TRAINING COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"{'=' * 65}")
    logger.info("  Checkpoints saved:")
    logger.info(f"    RGB Encoder:   sdc_project/outputs/checkpoints/best_model.pth")
    logger.info(f"    Depth Encoder: depth_encoder_share/depth_encoder/checkpoints/best.pt")
    logger.info(f"    OCTA + CLIP:   member1_text_pipeline/models/checkpoints/best_octa.pth")
    logger.info(f"    GACM Fusion:   multimodal_fusion_pipeline/outputs/checkpoints/best_fusion_model_finetuned.pth")
