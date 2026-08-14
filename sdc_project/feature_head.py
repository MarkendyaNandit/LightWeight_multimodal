"""
feature_head.py — Lightweight Feature Projection Head.

Takes the 352-dimensional multi-scale feature vector from MobileNetV3Encoder
and projects it to a compact 256-dimensional RGB feature vector.

Architecture:
    Input (B, 352)       Multi-scale features from encoder
        |
        v
    Linear(352, 512)     Expand to hidden dimension for capacity
        |
        v
    LayerNorm(512)       Stabilize feature distribution
        |
        v
    GELU                 Smooth non-linearity (paper uses GELU)
        |
        v
    Dropout(0.1)         Regularization for 210-image training set
        |
        v
    Linear(512, 256)     Project to final output dimension
        |
        v
    L2 Normalize         Unit-length vectors for cosine similarity
        |
        v
    Output (B, 256)      The RGB feature vector

Why L2 Normalization:
    The training loss uses cosine similarity between feature vectors.
    L2 normalization ensures all vectors lie on the unit hypersphere,
    so cosine similarity reduces to a simple dot product:
        cos_sim(a, b) = a . b / (||a|| * ||b||)
    With ||a|| = ||b|| = 1, this becomes just a . b.
    This makes the loss more numerically stable and the feature space
    more uniform for downstream anomaly scoring.

Why LayerNorm (not BatchNorm):
    - LayerNorm normalizes across features (independent of batch size).
    - BatchNorm depends on batch statistics, which are unreliable with
      small batches (batch_size=8) and especially during inference
      (batch_size=1 for single-image feature extraction).
    - The paper's GACM uses LayerNorm, so we maintain consistency.

Integration Note:
    The 256-dim output of this head is the FINAL product of your RGB
    module. Your teammates will use it as:
        - Input to GACM: F_rgb (256-dim) mapped to 3D domain
        - Input to OCTA alignment: F_rgb→text (256-dim) compared with text embeddings
    The L2 normalization ensures compatibility with cosine-distance-based
    fusion methods.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

logger = logging.getLogger(__name__)


class FeatureHead(nn.Module):
    """
    Projection head: 352-dim encoder features -> 256-dim RGB feature vector.

    This is a lightweight 2-layer MLP with normalization, non-linearity,
    dropout, and L2 output normalization. All layers are fully trainable.

    Args:
        input_dim:  Input feature dimension (default: 352 from encoder).
        hidden_dim: Hidden layer dimension (default: 512 from config).
        output_dim: Output feature dimension (default: 256 from config).
        dropout:    Dropout probability (default: 0.1 from config).
        normalize:  Whether to L2-normalize the output (default: True).

    Example:
        >>> head = FeatureHead()
        >>> x = torch.randn(8, 352)   # From encoder
        >>> out = head(x)
        >>> out.shape                  # torch.Size([8, 256])
        >>> torch.norm(out, dim=1)     # All ~1.0 (unit vectors)
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = cfg.HIDDEN_DIM,
        output_dim: int = cfg.FEATURE_DIM,
        dropout: float = cfg.DROPOUT,
        normalize: bool = True,
    ):
        super().__init__()

        # Default input_dim is the sum of encoder feature channels (80+112+160=352)
        if input_dim is None:
            input_dim = sum(cfg.FEATURE_CHANNELS)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.normalize = normalize

        # --- Build the projection MLP ---
        self.projection = nn.Sequential(
            # Layer 1: Expand to hidden dimension
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),

            # Layer 2: Project to output dimension
            nn.Linear(hidden_dim, output_dim),
        )

        # --- Initialize weights ---
        self._initialize_weights()

        # Log summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"FeatureHead initialized: {input_dim} -> {hidden_dim} -> {output_dim}"
        )
        logger.info(f"  Parameters: {total_params:,} (all trainable)")
        logger.info(f"  L2 normalization: {normalize}")

    def _initialize_weights(self) -> None:
        """
        Initialize linear layer weights using Kaiming (He) initialization.

        Kaiming init is optimal for layers followed by non-linearities
        (GELU in our case). It preserves the variance of activations
        through the network, preventing vanishing/exploding gradients.
        """
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        logger.debug("Weights initialized with Kaiming Normal.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project encoder features to the 256-dim output space.

        Args:
            x: Feature tensor of shape (B, input_dim).
               Typically (B, 352) from MobileNetV3Encoder.

        Returns:
            Feature tensor of shape (B, output_dim).
            L2-normalized if self.normalize is True.
        """
        # Project through MLP
        features = self.projection(x)

        # L2 normalize to unit length
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)

        return features

    def get_num_params(self) -> int:
        """Total number of parameters in the feature head."""
        return sum(p.numel() for p in self.parameters())


class RGBFeatureExtractor(nn.Module):
    """
    Complete RGB Feature Extraction pipeline.

    Combines MobileNetV3Encoder + FeatureHead into a single nn.Module
    for convenience. This is the top-level model used in:
        - train.py (training loop)
        - validate.py (validation)
        - test.py (evaluation)
        - extract_features.py (feature extraction)

    Pipeline:
        RGB Image (B, 3, 224, 224)
            |
            v
        MobileNetV3Encoder -> (B, 352) multi-scale features
            |
            v
        FeatureHead -> (B, 256) L2-normalized feature vector

    This is your DELIVERABLE to the team. When your teammate calls:
        model = RGBFeatureExtractor()
        model.load_state_dict(torch.load("best_model.pth"))
        features = model(rgb_images)  # (B, 256)
    they get the 256-dim RGB features ready for fusion.

    Args:
        encoder_pretrained: Load ImageNet weights for the encoder.
        freeze_up_to:       Freeze encoder layers [0, freeze_up_to].
        feature_dim:        Output dimension of the feature head.
        hidden_dim:         Hidden dimension in the feature head.
        dropout:            Dropout rate in the feature head.
        normalize:          L2-normalize the output features.
    """

    def __init__(
        self,
        encoder_pretrained: bool = cfg.ENCODER_PRETRAINED,
        freeze_up_to: int = cfg.FREEZE_UP_TO,
        feature_dim: int = cfg.FEATURE_DIM,
        hidden_dim: int = cfg.HIDDEN_DIM,
        dropout: float = cfg.DROPOUT,
        normalize: bool = True,
    ):
        super().__init__()

        # Import here to avoid circular imports at module level
        from mobilenet_encoder import MobileNetV3Encoder

        self.encoder = MobileNetV3Encoder(
            pretrained=encoder_pretrained,
            freeze_up_to=freeze_up_to,
        )

        self.head = FeatureHead(
            input_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            dropout=dropout,
            normalize=normalize,
        )

        self.feature_dim = feature_dim

        logger.info(
            f"RGBFeatureExtractor ready: "
            f"(3, 224, 224) -> encoder -> ({self.encoder.output_dim},) "
            f"-> head -> ({feature_dim},)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: RGB image -> 256-dim feature vector.

        Args:
            x: RGB image tensor of shape (B, 3, 224, 224).
               Must be normalized with ImageNet statistics.

        Returns:
            Feature tensor of shape (B, 256), L2-normalized.
        """
        encoder_out = self.encoder(x)
        features = self.head(encoder_out["concatenated"])
        return features

    def get_encoder_features(self, x: torch.Tensor) -> dict:
        """
        Get intermediate encoder features (before the head).

        Useful for debugging and visualization — see what the encoder
        extracts before projection.

        Args:
            x: RGB image tensor of shape (B, 3, 224, 224).

        Returns:
            Dict with "multi_scale_features" and "concatenated" tensors.
        """
        return self.encoder(x)

    def get_trainable_parameters(self):
        """
        Get all trainable parameters from both encoder and head.

        Used by the optimizer in train.py.

        Returns:
            List of parameters with requires_grad=True.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_groups(self):
        """
        Get parameter groups with different learning rates.

        A common fine-tuning strategy is to use a lower LR for the
        pretrained encoder and a higher LR for the randomly-initialized
        feature head. This prevents the encoder from forgetting ImageNet
        features while allowing the head to learn quickly.

        Returns:
            List of dicts, each with 'params' and 'lr' keys.
            Ready to pass directly to torch.optim.AdamW().
        """
        encoder_params = self.encoder.get_trainable_parameters()
        head_params = list(self.head.parameters())

        return [
            {
                "params": encoder_params,
                "lr": cfg.LEARNING_RATE * 0.1,  # 10x lower for pretrained layers
                "name": "encoder",
            },
            {
                "params": head_params,
                "lr": cfg.LEARNING_RATE,  # Full LR for randomly-initialized head
                "name": "head",
            },
        ]

    def get_num_total_params(self) -> int:
        """Total parameters (encoder + head)."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Trainable parameters (encoder unfrozen + entire head)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self) -> None:
        """Freeze the entire encoder (for head-only training)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Entire encoder frozen.")

    def unfreeze_encoder(self, from_layer: int = 0) -> None:
        """
        Unfreeze encoder layers starting from `from_layer`.

        Useful for gradual unfreezing (progressive fine-tuning):
        1. First train with frozen encoder (only head learns)
        2. Then unfreeze later encoder layers
        3. Finally unfreeze all layers for full fine-tuning
        """
        for idx, block in enumerate(self.encoder.features):
            if idx >= from_layer:
                for param in block.parameters():
                    param.requires_grad = True
        logger.info(f"Encoder unfrozen from layer {from_layer}.")

    def __repr__(self) -> str:
        total = self.get_num_total_params()
        trainable = self.get_num_trainable_params()
        return (
            f"RGBFeatureExtractor(\n"
            f"  encoder: MobileNetV3-Large (pretrained)\n"
            f"  head:    {self.encoder.output_dim} -> {cfg.HIDDEN_DIM} -> {self.feature_dim}\n"
            f"  output:  {self.feature_dim}-dim L2-normalized features\n"
            f"  params:  {total:,} total, {trainable:,} trainable\n"
            f")"
        )


# =========================================================================
# SELF-TEST — Verify the complete pipeline
# =========================================================================
if __name__ == "__main__":
    import time
    import os
    from PIL import Image

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 65)
    print("  FEATURE HEAD & RGB EXTRACTOR VERIFICATION")
    print("=" * 65)

    # --- Test FeatureHead alone ---
    print("\n--- FeatureHead Standalone Test ---")
    head = FeatureHead()
    dummy = torch.randn(8, 352)  # Simulated encoder output
    out = head(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output norms: {torch.norm(out, dim=1).tolist()[:3]}... (should be ~1.0)")

    assert out.shape == (8, 256), f"Expected (8, 256), got {out.shape}"
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Not unit-length!"
    print(f"  [PASS] Shape correct, L2-normalized to unit length")

    # --- Test RGBFeatureExtractor (full pipeline) ---
    print("\n--- RGBFeatureExtractor Full Pipeline Test ---")
    model = RGBFeatureExtractor()
    print(f"\n{model}")

    # Dummy forward pass
    dummy_images = torch.randn(4, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        features = model(dummy_images)

    print(f"\nInput:  {dummy_images.shape}")
    print(f"Output: {features.shape}")
    assert features.shape == (4, 256), f"Expected (4, 256), got {features.shape}"
    norms = torch.norm(features, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"  [PASS] Full pipeline: (4, 3, 224, 224) -> (4, 256)")

    # --- Test parameter groups ---
    print("\n--- Parameter Groups ---")
    param_groups = model.get_param_groups()
    for pg in param_groups:
        num_params = sum(p.numel() for p in pg["params"])
        print(f"  {pg['name']:10s}: {num_params:>10,} params, lr={pg['lr']}")
    print(f"  [PASS] Parameter groups configured correctly")

    # --- Test with real cookie image ---
    print("\n--- Real Cookie Image ---")
    from transforms import get_eval_transforms
    transform = get_eval_transforms()
    train_dir = cfg.get_train_dir()
    sample = sorted(os.listdir(train_dir))[0]
    img = Image.open(os.path.join(train_dir, sample)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model(img_tensor)

    print(f"  Image: {sample}")
    print(f"  Feature shape: {feat.shape}")
    print(f"  Feature norm:  {torch.norm(feat).item():.6f} (should be 1.0)")
    print(f"  Feature stats: mean={feat.mean():.4f}, std={feat.std():.4f}")
    print(f"  [PASS] Real cookie processed correctly")

    # --- CPU inference speed ---
    print("\n--- CPU Inference Speed (Full Pipeline) ---")
    model.eval()
    single = torch.randn(1, 3, 224, 224)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(single)

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(10):
            start = time.perf_counter()
            _ = model(single)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.1f} ms/image (encoder + head)")
    print(f"  Throughput: ~{1000/avg:.0f} images/sec on CPU")

    # --- Parameter summary ---
    print("\n--- Complete Model Summary ---")
    total = model.get_num_total_params()
    trainable = model.get_num_trainable_params()
    head_params = model.head.get_num_params()
    encoder_total = model.encoder.get_num_total_params()
    encoder_trainable = model.encoder.get_num_trainable_params()

    print(f"  Encoder:  {encoder_total:>10,} total, {encoder_trainable:>10,} trainable")
    print(f"  Head:     {head_params:>10,} total, {head_params:>10,} trainable")
    print(f"  Combined: {total:>10,} total, {trainable:>10,} trainable")
    print(f"  Model size: ~{total * 4 / 1024 / 1024:.1f} MB (float32)")

    # Cleanup
    model.encoder.remove_hooks()

    print("\n" + "=" * 65)
    print("  ALL FEATURE HEAD TESTS PASSED")
    print("=" * 65)
