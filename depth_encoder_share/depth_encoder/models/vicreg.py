"""
models/vicreg.py

VICReg (Variance-Invariance-Covariance Regularisation) SSL wrapper.

Architecture during training:
  DepthEncoder (backbone + 256-d projection head)
       ↓
  Expander MLP  (256 → 2048)  ← used ONLY for computing VICReg loss
       ↓
  VICReg loss

At inference / feature extraction, only the DepthEncoder (256-d output)
is used.  The expander is discarded after training.

Reference:
  Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
  for Self-Supervised Learning", ICLR 2022.
  https://arxiv.org/abs/2105.04906
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .depth_encoder import DepthEncoder


# ---------------------------------------------------------------------------
# Expander MLP
# ---------------------------------------------------------------------------

class Expander(nn.Module):
    """
    Three-layer MLP that maps 256-d embeddings to a higher-dimensional
    space (default 2048) for the VICReg loss computation.

    The expander is *not* used at inference — only the DepthEncoder output
    (256-d) is the final embedding.

    Architecture:
      Linear(in, dim) → BN → ReLU → Linear(dim, dim) → BN → ReLU → Linear(dim, dim)
    """

    def __init__(self, in_dim: int = 256, out_dim: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# VICReg Model
# ---------------------------------------------------------------------------

class VICRegModel(nn.Module):
    """
    VICReg self-supervised learning wrapper around DepthEncoder.

    During training:
      1. Two augmented views (v1, v2) are fed through the shared encoder.
      2. Both 256-d embeddings are expanded to `expander_dim` dimensions.
      3. The VICReg loss is computed on the expanded representations.

    At inference:
      - Call `encoder.forward(x)` directly to get the 256-d embedding.
      - The expander is not needed and can be ignored.

    Parameters
    ----------
    encoder : DepthEncoder
        The depth encoder to train.
    expander_dim : int
        Dimension of the expander MLP output (default 2048).
    """

    def __init__(
        self,
        encoder: DepthEncoder,
        expander_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.encoder  = encoder
        self.expander = Expander(
            in_dim=encoder.embedding_dim,
            out_dim=expander_dim,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for SSL training.

        Parameters
        ----------
        view1, view2 : torch.Tensor
            Two augmented views, each (B, 3, H, W).

        Returns
        -------
        z1, z2 : torch.Tensor
            Expanded representations, each (B, expander_dim).
            Pass these to `vicreg_loss(z1, z2, ...)`.
        """
        emb1, _ = self.encoder(view1)   # (B, 256) — feature_maps discarded during train
        emb2, _ = self.encoder(view2)   # (B, 256)

        z1 = self.expander(emb1)        # (B, 2048)
        z2 = self.expander(emb2)        # (B, 2048)

        return z1, z2

    # ------------------------------------------------------------------
    @property
    def backbone_parameters(self):
        """Parameters of the backbone (for LR scheduling)."""
        return self.encoder.backbone.parameters()

    @property
    def head_parameters(self):
        """Parameters of the projection head + expander."""
        return list(self.encoder.projection_head.parameters()) + \
               list(self.expander.parameters())
