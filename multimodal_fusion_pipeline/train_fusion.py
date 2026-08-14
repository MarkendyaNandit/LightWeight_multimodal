"""
train_fusion.py — Phase 9: Training GACM and Visual Fusion Model.

Trains the VisualFusionPipeline on normal cookie RGB+Depth paired features.
Learns a compact visual representation (F_vis) using self-supervised 
consistency + compactness loss.
"""

import os
import sys
import time
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from gacm import LightweightGACM
from fusion_model import VisualFusionPipeline
from dataset_fusion import load_feature_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

class VisualFusionLoss(nn.Module):
    """
    Self-supervised loss for visual fusion.
    - Consistency: F_vis should be consistent with both RGB and Depth modalities.
    - Compactness: Normal F_vis vectors cluster around a running center vector.
    """
    def __init__(self, dim=cfg.FEATURE_DIM):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.register_buffer("_center_ema", torch.zeros(1, dim))
        self.ema_decay = 0.99

    def forward(self, f_vis, f_rgb, f_depth):
        # 1. Cosine Consistency with inputs
        cos_rgb = (1.0 - F.cosine_similarity(f_vis, f_rgb, dim=1)).mean()
        cos_depth = (1.0 - F.cosine_similarity(f_vis, f_depth, dim=1)).mean()
        consistency_loss = (cos_rgb + cos_depth) / 2.0

        # 2. Compactness Loss to center
        center = self.center.detach()
        compactness_loss = ((f_vis - center) ** 2).sum(dim=1).mean()

        # Update center via EMA
        with torch.no_grad():
            batch_mean = f_vis.mean(dim=0, keepdim=True)
            self._center_ema = self.ema_decay * self._center_ema + (1.0 - self.ema_decay) * batch_mean
            self.center.copy_(self._center_ema)

        total_loss = consistency_loss + 0.5 * compactness_loss
        return total_loss, consistency_loss.item(), compactness_loss.item()

def train():
    device = cfg.DEVICE
    logger.info("=" * 65)
    logger.info("  PHASE 9 — VISUAL FUSION & GACM TRAINING")
    logger.info("=" * 65)
    logger.info(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, _ = load_feature_splits()

    # 2. Build Model & Loss
    model = VisualFusionPipeline(dim=cfg.FEATURE_DIM, hidden_dim=cfg.HIDDEN_DIM).to(device)
    loss_fn = VisualFusionLoss(dim=cfg.FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    best_val_loss = float("inf")

    # 3. Training Loop
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for f_rgb, f_depth in train_loader:
            f_rgb = f_rgb.to(device)
            f_depth = f_depth.to(device)

            f_vis = model(f_rgb, f_depth)
            loss, c_loss, comp_loss = loss_fn(f_vis, f_rgb, f_depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_loss = total_loss / batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for f_rgb, f_depth in val_loader:
                f_rgb = f_rgb.to(device)
                f_depth = f_depth.to(device)
                f_vis = model(f_rgb, f_depth)
                l, _, _ = loss_fn(f_vis, f_rgb, f_depth)
                val_loss += l.item()
                val_batches += 1
        val_loss /= max(val_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{cfg.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_fusion_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss_fn_state_dict": loss_fn.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

    elapsed = time.time() - start_time
    logger.info(f"\nTraining Complete in {elapsed:.1f}s")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {os.path.join(cfg.CHECKPOINT_DIR, 'best_fusion_model.pth')}")

if __name__ == "__main__":
    train()
