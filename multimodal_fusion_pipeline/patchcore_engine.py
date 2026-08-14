"""
patchcore_engine.py — Multimodal PatchCore Anomaly Detection Engine

Implements patch-level feature extraction, local neighborhood aggregation,
coreset memory bank management, anomaly scoring, and 2D heatmap generation.
Designed for 2mm micro-defect detection while keeping memory lightweight (< 300MB RAM).
"""

import os, sys, math, time, logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

logger = logging.getLogger("PatchCoreEngine")

# ── Local Neighborhood Aggregation ──────────────────────────────────
class PatchNeighborhoodAggregator(nn.Module):
    """
    Applies 3x3 average pooling over spatial grid to incorporate
    local spatial context into each patch vector (standard PatchCore step).
    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

# ── Patch Feature Extractor ─────────────────────────────────────────
class MultimodalPatchExtractor(nn.Module):
    """
    Extracts spatial patch feature maps from RGB and Depth encoders.
    Produces (B, H*W, D) patch vectors without global spatial pooling.
    """
    def __init__(self, rgb_model, depth_model, fusion_model=None, target_size: Tuple[int, int] = (14, 14)):
        super().__init__()
        self.rgb_model    = rgb_model
        self.depth_model  = depth_model
        self.fusion_model = fusion_model
        self.target_size  = target_size
        self.aggregator   = PatchNeighborhoodAggregator(kernel_size=3, stride=1, padding=1)

    @torch.no_grad()
    def extract_patch_features(self, rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input:  rgb_tensor   (B, 3, 224, 224)
                depth_tensor (B, 3, 224, 224)
        Output: patch_vectors (B, H*W, D) where H=14, W=14, D=256
        """
        self.rgb_model.eval()
        self.depth_model.eval()

        # Extract unpooled intermediate feature maps from RGB encoder
        # MobileNetV3 blocks 7, 11, 13 produce features before global pooling
        rgb_feats = []
        x_rgb = rgb_tensor
        for i, block in enumerate(self.rgb_model.encoder.features):
            x_rgb = block(x_rgb)
            if i in [7, 11, 13]:
                rgb_feats.append(x_rgb)

        # Resize all multi-scale feature maps to target_size (14, 14) and concatenate
        resized_rgb = []
        for feat in rgb_feats:
            if feat.shape[-2:] != self.target_size:
                feat = F.interpolate(feat, size=self.target_size, mode="bilinear", align_corners=False)
            resized_rgb.append(feat)

        # Concatenate along channel dimension: (B, C_rgb, 14, 14)
        rgb_spatial = torch.cat(resized_rgb, dim=1)

        # Apply 3x3 local neighborhood aggregation
        rgb_spatial = self.aggregator(rgb_spatial)

        # Process depth tensor similarly
        x_dep = depth_tensor
        depth_spatial = None
        if hasattr(self.depth_model, "encoder") and hasattr(self.depth_model.encoder, "features"):
            dep_feats = []
            for i, block in enumerate(self.depth_model.encoder.features):
                x_dep = block(x_dep)
                if i in [7, 11, 13]:
                    dep_feats.append(x_dep)
            resized_dep = []
            for feat in dep_feats:
                if feat.shape[-2:] != self.target_size:
                    feat = F.interpolate(feat, size=self.target_size, mode="bilinear", align_corners=False)
                resized_dep.append(feat)
            depth_spatial = torch.cat(resized_dep, dim=1)
            depth_spatial = self.aggregator(depth_spatial)
        else:
            # Fallback if depth encoder structure differs
            depth_spatial = torch.zeros_like(rgb_spatial)

        # Pass through RGB & Depth feature heads to get 256-d representations
        if hasattr(self.rgb_model, "head"):
            # Reshape (B, C, H, W) -> (B*H*W, C)
            B, C_r, H, W = rgb_spatial.shape
            rgb_flat = rgb_spatial.permute(0, 2, 3, 1).reshape(-1, C_r)
            rgb_proj = self.rgb_model.head(rgb_flat).reshape(B, H, W, 256).permute(0, 3, 1, 2)
        else:
            rgb_proj = F.adaptive_avg_pool2d(rgb_spatial, (14, 14))

        if hasattr(self.depth_model, "head"):
            B, C_d, H, W = depth_spatial.shape
            dep_flat = depth_spatial.permute(0, 2, 3, 1).reshape(-1, C_d)
            dep_proj = self.depth_model.head(dep_flat).reshape(B, H, W, 256).permute(0, 3, 1, 2)
        else:
            dep_proj = rgb_proj

        # Fuse RGB & Depth spatial maps using GACM if provided
        if self.fusion_model is not None:
            B, C, H, W = rgb_proj.shape
            fr_flat = rgb_proj.permute(0, 2, 3, 1).reshape(B * H * W, C)
            fd_flat = dep_proj.permute(0, 2, 3, 1).reshape(B * H * W, C)
            fused_flat = self.fusion_model(fr_flat, fd_flat)
            fused_spatial = fused_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            fused_spatial = 0.5 * (rgb_proj + dep_proj)

        # Normalize features along channel dimension: L2 norm = 1
        fused_spatial = F.normalize(fused_spatial, p=2, dim=1)

        # Reshape to (B, H*W, D) = (B, 196, 256)
        B, C, H, W = fused_spatial.shape
        patch_vectors = fused_spatial.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return patch_vectors

# ── K-Center Greedy Coreset Selection ───────────────────────────────
class KCenterGreedyCoreset:
    """
    Subsamples a large patch memory bank (e.g. 50,000 vectors) down to a
    compact coreset (e.g. 2,000 vectors) using greedy K-Center selection.
    Ensures maximum coverage of normal feature space while saving 90% memory.
    """
    def __init__(self, sampling_ratio: float = 0.10):
        self.sampling_ratio = sampling_ratio

    def sample(self, memory_bank: torch.Tensor, max_coreset_size: int = 15000) -> torch.Tensor:
        """
        memory_bank: (N, D) tensor on CPU/GPU
        returns:     (K, D) coreset tensor where K = min(max_coreset_size, max(100, int(N * sampling_ratio)))
        """
        N, D = memory_bank.shape
        K = max(100, min(N, int(N * self.sampling_ratio)))
        K = min(K, max_coreset_size)  # Cap size to prevent eternal loops

        if K >= N:
            return memory_bank

        logger.info(f"    Subsampling coreset: {N} patches -> {K} patches (capped at {max_coreset_size})")

        # Normalize memory bank for L2 distance calculation
        memory_bank = F.normalize(memory_bank.float(), p=2, dim=1)
        
        # Use GPU for extremely fast distance calculations if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mb_device = memory_bank.to(device)

        # Start with a random index
        selected_indices = [int(torch.randint(0, N, (1,)).item())]
        min_distances = torch.norm(mb_device - mb_device[selected_indices[0]], dim=1)

        for _ in range(1, K):
            # Select the point farthest from current coreset
            farthest_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(farthest_idx)

            # Update minimum distances
            new_dist = torch.norm(mb_device - mb_device[farthest_idx], dim=1)
            min_distances = torch.minimum(min_distances, new_dist)

        return memory_bank[selected_indices].cpu()

# Smooth/manufactured surfaces: use strict MAX patch (catches single-patch micro-defects)
SMOOTH_CATEGORIES = {"phone_screen", "car_metal", "pcb"}
# Organic/textured surfaces: use Top-K average (smooths natural texture noise)
ORGANIC_CATEGORIES = set()
TOP_K = 10  # Average the 10 highest patch distances for organic categories

# ── PatchCore Memory Bank & Inference Pipeline ──────────────────────
class PatchCorePipeline:
    """
    Manages per-category coreset memory banks, computes patch-level anomaly scores,
    and generates 2D anomaly heatmaps for web visualization.
    Uses HYBRID scoring: MAX for smooth categories, Top-K avg for organic categories.
    """
    def __init__(self, extractor: MultimodalPatchExtractor, coreset_banks: Dict[str, torch.Tensor] = None):
        self.extractor     = extractor
        self.coreset_banks = coreset_banks if coreset_banks is not None else {}

    def set_coreset_bank(self, category: str, bank: torch.Tensor):
        self.coreset_banks[category] = bank.float()

    @torch.no_grad()
    def score_image(self, rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor, category: str) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Input:  rgb_tensor   (1, 3, 224, 224)
                depth_tensor (1, 3, 224, 224)
                category     str
        Output:
                anomaly_score (float): hybrid score (MAX for smooth, Top-K avg for organic)
                patch_map     (14, 14 np.ndarray): 2D patch distance map
                heatmap       (224, 224 np.ndarray): smooth bicubic heat map [0..255]
        """
        if category not in self.coreset_banks:
            raise ValueError(f"No coreset memory bank loaded for category '{category}'")

        # Extract 196 patch vectors: (1, 196, 256)
        patch_vecs = self.extractor.extract_patch_features(rgb_tensor, depth_tensor)
        patch_vecs = patch_vecs.squeeze(0)  # (196, 256)

        coreset = self.coreset_banks[category].to(patch_vecs.device)  # (K, 256)

        # Compute L2 distance matrix: (196, K)
        dist_matrix = torch.cdist(patch_vecs, coreset, p=2.0)  # (196, K)

        # For each query patch, find distance to closest coreset patch
        min_patch_dists, _ = torch.min(dist_matrix, dim=1)  # (196,)

        # HYBRID scoring: select method based on category type
        if category in ORGANIC_CATEGORIES:
            # Top-K average: smooth out natural texture noise
            k = min(TOP_K, min_patch_dists.shape[0])
            topk_vals, _ = torch.topk(min_patch_dists, k)
            anomaly_score = float(torch.mean(topk_vals).item())
        else:
            # MAX patch: catch single-patch micro-defects on smooth surfaces
            anomaly_score = float(torch.max(min_patch_dists).item())

        # Reshape (196,) into 2D grid (14, 14)
        patch_map_tensor = min_patch_dists.reshape(14, 14)
        patch_map_np     = patch_map_tensor.cpu().numpy()

        # Generate smooth 224x224 heatmap using bicubic interpolation
        heatmap_tensor = F.interpolate(
            patch_map_tensor.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bicubic",
            align_corners=False
        ).squeeze()

        # Scale heatmap to uint8 [0..255]
        h_min, h_max = heatmap_tensor.min(), heatmap_tensor.max()
        h_norm = (heatmap_tensor - h_min) / (h_max - h_min + 1e-8)
        heatmap_uint8 = (h_norm * 255.0).cpu().numpy().astype(np.uint8)

        return anomaly_score, patch_map_np, heatmap_uint8

