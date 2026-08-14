"""
models/depth_encoder.py

MobileNetV3-Large depth encoder with:
  - ImageNet-pretrained backbone
  - Intermediate feature map extraction via forward hooks
  - Custom 256-dimensional projection head:
      Linear → BatchNorm1d → ReLU → Dropout → Linear → 256-d

The encoder is designed to be later combined with:
  - RGB MobileNetV3 encoder (identical architecture, separate weights)
  - GACM for RGB-to-depth feature alignment
  - CLIP text encoder
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import MobileNet_V3_Large_Weights


# ---------------------------------------------------------------------------
# DepthEncoder
# ---------------------------------------------------------------------------

class DepthEncoder(nn.Module):
    """
    MobileNetV3-Large backbone with a custom projection head.

    Architecture overview
    ---------------------
    Input: (B, 3, 224, 224) — 3-channel depth map (Z repeated × 3)

    Backbone:  MobileNetV3-Large features[0..16]
               ↓
    Intermediate feature maps tapped at configurable layer indices
               ↓
    AdaptiveAvgPool2d (1, 1)  [built into MobileNetV3 avgpool]
               ↓
    Projection head:
      Linear(960, hidden_dim)  →  BN1d  →  ReLU  →  Dropout  →  Linear(hidden_dim, 256)
               ↓
    Output: 256-dimensional embedding

    Parameters
    ----------
    embedding_dim : int
        Final embedding dimension (default 256).
    hidden_dim : int
        Hidden dimension of the projection head (default 512).
    dropout : float
        Dropout probability in the projection head (default 0.1).
    pretrained : bool
        Load ImageNet weights for the backbone (default True).
    feature_layers : List[int]
        Indices into `backbone.features` at which to capture intermediate
        feature maps.  These are stored in `self.feature_maps` after each
        forward pass.
    freeze_backbone : bool
        If True, backbone weights are frozen (useful for linear probe eval).
    """

    # MobileNetV3-Large final channel count before the classifier
    _BACKBONE_OUT_CHANNELS = 960

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
        feature_layers: Optional[List[int]] = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.feature_layer_indices = feature_layers or [3, 7, 13]
        self.feature_maps: Dict[int, torch.Tensor] = {}

        # ----------------------------------------------------------------
        # Backbone
        # ----------------------------------------------------------------
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        _mv3 = tvm.mobilenet_v3_large(weights=weights)

        # Keep only the feature extractor (discard the original classifier)
        self.backbone = _mv3.features   # nn.Sequential of 17 blocks
        self.avgpool  = _mv3.avgpool    # AdaptiveAvgPool2d → (B, 960, 1, 1)

        # Register forward hooks to capture intermediate feature maps
        self._hooks: list = []
        self._register_feature_hooks()

        # ----------------------------------------------------------------
        # Projection head
        # ----------------------------------------------------------------
        self.projection_head = nn.Sequential(
            nn.Linear(self._BACKBONE_OUT_CHANNELS, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embedding_dim, bias=True),
        )

        # ----------------------------------------------------------------
        # Optional backbone freeze
        # ----------------------------------------------------------------
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Weight initialisation for the projection head
        self._init_projection_head()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _hook_fn(self, layer_idx: int):
        """Create a hook closure that captures the output of layer `layer_idx`."""
        def hook(module: nn.Module, inp, output: torch.Tensor):
            self.feature_maps[layer_idx] = output.detach()
        return hook

    def _register_feature_hooks(self) -> None:
        """Attach forward hooks to the requested backbone layers."""
        for idx in self.feature_layer_indices:
            if idx >= len(self.backbone):
                raise ValueError(
                    f"feature_layer index {idx} out of range; "
                    f"backbone has {len(self.backbone)} layers (0–{len(self.backbone)-1})."
                )
            h = self.backbone[idx].register_forward_hook(self._hook_fn(idx))
            self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks (call before serialising model)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_projection_head(self) -> None:
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 3, H, W).

        Returns
        -------
        embedding : torch.Tensor
            Shape (B, 256) — L2-normalised 256-d feature vector.
        feature_maps : Dict[int, torch.Tensor]
            Intermediate feature maps keyed by backbone layer index.
            Populated as a side-effect by the registered hooks.
        """
        # Clear previous feature maps
        self.feature_maps.clear()

        # Backbone forward (hooks fire automatically)
        x = self.backbone(x)          # (B, 960, 7, 7)
        x = self.avgpool(x)           # (B, 960, 1, 1)
        x = x.flatten(1)              # (B, 960)

        # Projection head
        embedding = self.projection_head(x)   # (B, 256)

        # L2 normalise for downstream cosine-similarity / alignment
        embedding = nn.functional.normalize(embedding, dim=1)

        return embedding, dict(self.feature_maps)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_feature_channels(self) -> Dict[int, int]:
        """
        Return the number of output channels for each tapped feature layer.
        Useful for the GACM module to know input dimensions.
        """
        # MobileNetV3-Large channel widths at each features index:
        _channels = {
            0: 16,  1: 16,  2: 24,  3: 24,  4: 40,
            5: 40,  6: 40,  7: 80,  8: 80,  9: 80,
            10: 80, 11: 112, 12: 112, 13: 160, 14: 160,
            15: 160, 16: 960,
        }
        return {idx: _channels[idx] for idx in self.feature_layer_indices}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DepthEncoder("
            f"embedding_dim={self.embedding_dim}, "
            f"feature_layers={self.feature_layer_indices})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_encoder(cfg: dict) -> DepthEncoder:
    """
    Construct a DepthEncoder from the model config block.

    Parameters
    ----------
    cfg : dict
        The `model` block from config.yaml.

    Returns
    -------
    DepthEncoder
    """
    return DepthEncoder(
        embedding_dim=cfg.get("embedding_dim", 256),
        hidden_dim=cfg.get("projection_hidden_dim", 512),
        dropout=cfg.get("dropout", 0.1),
        pretrained=cfg.get("pretrained", True),
        feature_layers=cfg.get("feature_layers", [3, 7, 13]),
    )
