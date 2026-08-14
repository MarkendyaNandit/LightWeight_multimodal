"""
fusion_model.py — Phase 6: Visual Fusion Pipeline Model.

Combines RGB (256-D) and Depth (256-D) feature vectors via GACM into a 
unified 256-D Visual Feature Vector (F_vis).
"""

import torch
import torch.nn as nn
from gacm import LightweightGACM
from config import cfg

class VisualFusionPipeline(nn.Module):
    """
    Top-level Visual Fusion Model.
    
    Takes:
        f_rgb:   (B, 256) RGB feature vector
        f_depth: (B, 256) Depth feature vector

    Returns:
        f_vis:   (B, 256) Fused visual feature vector (L2-normalized)
    """

    def __init__(self, dim: int = cfg.FEATURE_DIM, hidden_dim: int = cfg.HIDDEN_DIM):
        super().__init__()
        self.gacm = LightweightGACM(dim=dim, hidden_dim=hidden_dim)

    def forward(self, f_rgb: torch.Tensor, f_depth: torch.Tensor) -> torch.Tensor:
        return self.gacm(f_rgb, f_depth)
