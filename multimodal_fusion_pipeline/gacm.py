"""
gacm.py — Phase 7: Lightweight Geometry-Aware Cross-Modal Mapper.

Bifurcates RGB features into semantic and geometric branches, uses a 
sigmoid geometry-prior gate (GPG) driven by Depth structural features,
and outputs a geometry-calibrated visual representation (F_vis).

Based on Section III-A of the research paper (arXiv 2604.22899) and 
Phase 7 of the DOCX roadmap guide.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightGACM(nn.Module):
    """
    Geometry-Aware Cross-Modal Mapper (GACM).

    Input:
        F_rgb:   (B, 256) RGB feature vector
        F_depth: (B, 256) Depth feature vector

    Output:
        F_vis:   (B, 256) Fused geometry-aware visual feature vector (L2-normalized)
    """

    def __init__(self, dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim

        # 1. Branch Bifurcation: Semantic vs Geometric
        self.phi_sem = nn.Linear(dim, dim)
        self.phi_geo = nn.Linear(dim, dim)

        # 2. Geometry Prior Gating (GPG)
        # Combines geometry branch and depth map structural features into a sigmoid gate
        self.w_gate = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

        # 3. Non-linear Mimicry Network
        self.mimicry_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

        # 4. Residual Projection
        self.residual_proj = nn.Linear(dim, dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, f_rgb: torch.Tensor, f_depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        f_rgb: (B, 256)
        f_depth: (B, 256)
        """
        # Step 1: Bifurcate RGB features into semantic and geometric components
        f_sem = self.phi_sem(f_rgb)  # (B, 256)
        f_geo = self.phi_geo(f_rgb)  # (B, 256)

        # Step 2: Compute Geometry Prior Gate (G)
        gate_input = torch.cat([f_geo, f_depth], dim=1)  # (B, 512)
        g = self.sigmoid(self.w_gate(gate_input))        # (B, 256)

        # Step 3: Adaptive Fusion
        f_fused = f_geo * g + f_sem * (1.0 - g)          # (B, 256)

        # Step 4: Non-linear Mimicry + Residual Connection
        f_mapped = self.mimicry_net(f_fused) + self.residual_proj(f_rgb)

        # Step 5: L2 Normalization
        f_vis = F.normalize(f_mapped, p=2, dim=1)         # (B, 256)

        return f_vis
