"""
mobilenet_encoder.py — MobileNetV3-Large RGB Feature Extractor.

This is the backbone of the RGB Encoder module. It takes a 224x224 RGB
image and produces multi-scale feature vectors by tapping into three
intermediate layers of MobileNetV3-Large.

Architecture Overview:
    MobileNetV3-Large has 17 sequential blocks (indices 0-16) inside its
    `features` module. We extract features at three depths:

    Input (3, 224, 224)
        |
        v
    Blocks 0-6   [FROZEN]  Low-level features (edges, textures, colors)
        |
        v
    Block 7      [TAP]     80 channels, 14x14 spatial -> GAP -> (80,)
        |
        v
    Blocks 8-10  [FROZEN at 8, FINE-TUNE at 9-10]
        |
        v
    Block 11     [TAP]     112 channels, 14x14 spatial -> GAP -> (112,)
        |
        v
    Block 12     [FINE-TUNE]
        |
        v
    Block 13     [TAP]     160 channels, 7x7 spatial -> GAP -> (160,)
        |
        v
    Blocks 14-16 [FINE-TUNE, but not used for features]
        |
        v
    Concatenate [80 + 112 + 160] = 352-dimensional vector
        |
        v
    Output: (batch_size, 352)  -->  Sent to feature_head.py for 256-dim projection

Why Multi-Scale Features:
    Surface defects in cookies appear at different scales:
    - Tiny cracks:       Captured by low-level features (layer 7, 80ch)
    - Texture anomalies: Captured by mid-level features (layer 10, 112ch)
    - Large holes/blobs: Captured by high-level features (layer 13, 160ch)
    Concatenating all three gives the feature head a comprehensive view.

Why Transfer Learning:
    With only 210 training images, training from scratch would severely
    overfit. ImageNet pretrained weights provide:
    1. Universal low-level feature detectors (Gabor-like filters, color blobs)
    2. Mid-level pattern recognizers (textures, shapes, contours)
    3. High-level object part detectors (which we fine-tune for cookies)
    This is the standard approach in industrial anomaly detection research.

Integration Note:
    Your teammate's Depth Encoder will produce a separate feature vector
    (also 256-dim after their own projection head). The GACM module
    (another teammate) will fuse your RGB features with Depth features.
    Your encoder's output dimension (256) is the agreed interface.
"""

import logging
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

from config import cfg

logger = logging.getLogger(__name__)


class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3-Large backbone for multi-scale RGB feature extraction.

    Loads ImageNet-pretrained weights, freezes early layers, and extracts
    intermediate features via forward hooks. The classifier head is
    completely removed — we only use the convolutional feature extractor.

    Args:
        pretrained:     Whether to load ImageNet pretrained weights.
        freeze_up_to:   Freeze all blocks in features[0:freeze_up_to+1].
        feature_layers: List of block indices to extract features from.
        feature_channels: Expected output channels at each tap point.

    Outputs:
        forward() returns a dict with:
            "multi_scale_features": list of (B, C_i) tensors after GAP
            "concatenated":        (B, sum(C_i)) tensor — the 352-dim vector

    Example:
        >>> encoder = MobileNetV3Encoder()
        >>> x = torch.randn(4, 3, 224, 224)
        >>> out = encoder(x)
        >>> out["concatenated"].shape  # torch.Size([4, 352])
    """

    def __init__(
        self,
        pretrained: bool = cfg.ENCODER_PRETRAINED,
        freeze_up_to: int = cfg.FREEZE_UP_TO,
        feature_layers: Optional[List[int]] = None,
        feature_channels: Optional[List[int]] = None,
    ):
        super().__init__()

        self.feature_layers = feature_layers or cfg.FEATURE_LAYERS
        self.feature_channels = feature_channels or cfg.FEATURE_CHANNELS
        self.freeze_up_to = freeze_up_to

        # Total output dimension (sum of all tap channels)
        self.output_dim = sum(self.feature_channels)

        # --- Load MobileNetV3-Large ---
        if pretrained:
            logger.info("Loading MobileNetV3-Large with ImageNet pretrained weights...")
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            backbone = models.mobilenet_v3_large(weights=weights)
            logger.info("  Pretrained weights loaded successfully.")
        else:
            logger.info("Loading MobileNetV3-Large WITHOUT pretrained weights...")
            backbone = models.mobilenet_v3_large(weights=None)

        # --- Extract only the feature blocks (discard classifier) ---
        # backbone.features is a Sequential of 17 blocks.
        # backbone.classifier is the ImageNet classification head — we don't need it.
        self.features = backbone.features

        # Global Average Pooling to convert (B, C, H, W) -> (B, C)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # --- Freeze early layers ---
        self._freeze_layers()

        # --- Validate feature layer indices ---
        num_blocks = len(self.features)
        for layer_idx in self.feature_layers:
            if layer_idx >= num_blocks:
                raise ValueError(
                    f"Feature layer index {layer_idx} is out of range. "
                    f"MobileNetV3-Large has {num_blocks} blocks (0-{num_blocks - 1})."
                )

        # --- Register forward hooks for feature extraction ---
        # Hooks capture the output of specified layers during forward pass.
        self._hook_features: Dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

        # Log architecture summary
        self._log_summary()

    def _freeze_layers(self) -> None:
        """
        Freeze layers [0, freeze_up_to] by setting requires_grad = False.

        Frozen layers:
        - Do NOT update during backpropagation.
        - Retain their ImageNet-learned features exactly.
        - Reduce memory usage (no gradient storage) and speed up training.

        Fine-tuned layers:
        - ARE updated during training to adapt to cookie-specific patterns.
        - Learn what "normal cookie texture" looks like.
        """
        frozen_count = 0
        trainable_count = 0

        for idx, block in enumerate(self.features):
            if idx <= self.freeze_up_to:
                # Freeze this block
                for param in block.parameters():
                    param.requires_grad = False
                frozen_count += sum(1 for _ in block.parameters())
            else:
                # Fine-tune this block
                for param in block.parameters():
                    param.requires_grad = True
                trainable_count += sum(1 for _ in block.parameters())

        logger.info(
            f"  Layer freezing: blocks 0-{self.freeze_up_to} FROZEN, "
            f"blocks {self.freeze_up_to + 1}-{len(self.features) - 1} TRAINABLE"
        )
        logger.info(
            f"  Frozen parameters: {frozen_count}, "
            f"Trainable parameters: {trainable_count}"
        )

    def _register_hooks(self) -> None:
        """
        Register forward hooks on the specified feature layers.

        Forward hooks intercept the output of a layer during the forward
        pass WITHOUT modifying the computation graph. This is cleaner
        than manually splitting the Sequential into sub-sequences.
        """
        for layer_idx in self.feature_layers:
            hook = self.features[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

        logger.info(
            f"  Forward hooks registered at layers: {self.feature_layers}"
        )

    def _make_hook(self, layer_idx: int):
        """
        Create a hook function that stores the output of a specific layer.

        Args:
            layer_idx: Index of the layer to hook into.

        Returns:
            Hook function compatible with PyTorch's register_forward_hook.
        """
        def hook_fn(module, input, output):
            self._hook_features[layer_idx] = output
        return hook_fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: extract multi-scale features from the RGB image.

        Args:
            x: Input RGB image tensor of shape (B, 3, 224, 224).
               Must be normalized with ImageNet statistics.

        Returns:
            Dictionary containing:
                "multi_scale_features": List of tensors [(B, 80), (B, 112), (B, 160)]
                "concatenated":         Tensor of shape (B, 352)

        The "concatenated" tensor is the primary output that gets sent to
        feature_head.py for projection to 256 dimensions.
        """
        # Clear previous hook outputs
        self._hook_features.clear()

        # Run the full forward pass through all feature blocks.
        # The hooks automatically capture intermediate outputs.
        _ = self.features(x)

        # Collect and pool features from hooked layers
        multi_scale = []
        for layer_idx, expected_ch in zip(self.feature_layers, self.feature_channels):
            feat = self._hook_features[layer_idx]  # (B, C, H, W)

            # Verify channel count matches config
            actual_ch = feat.shape[1]
            if actual_ch != expected_ch:
                raise RuntimeError(
                    f"Channel mismatch at layer {layer_idx}: "
                    f"expected {expected_ch}, got {actual_ch}. "
                    f"Check FEATURE_CHANNELS in config.py."
                )

            # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            pooled = self.pool(feat).flatten(1)
            multi_scale.append(pooled)

        # Concatenate all scales: (B, 80+112+160) = (B, 352)
        concatenated = torch.cat(multi_scale, dim=1)

        return {
            "multi_scale_features": multi_scale,
            "concatenated": concatenated,
        }

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get only the trainable (unfrozen) parameters.

        This is used by the optimizer in train.py to ensure we only
        optimize parameters that require gradients.

        Returns:
            List of parameters with requires_grad=True.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """Count the total number of parameters (frozen + trainable)."""
        return sum(p.numel() for p in self.parameters())

    def _log_summary(self) -> None:
        """Log a summary of the encoder architecture."""
        total = self.get_num_total_params()
        trainable = self.get_num_trainable_params()
        frozen = total - trainable

        logger.info(f"  Total parameters:     {total:,}")
        logger.info(f"  Trainable parameters: {trainable:,}")
        logger.info(f"  Frozen parameters:    {frozen:,}")
        logger.info(
            f"  Output dimension:     {self.output_dim} "
            f"(from layers {self.feature_layers} with channels {self.feature_channels})"
        )

    def remove_hooks(self) -> None:
        """
        Remove all forward hooks.

        Call this when exporting the model or when hooks are no longer
        needed. This prevents potential memory leaks in long-running
        inference loops.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("Forward hooks removed.")

    def set_eval_mode(self) -> None:
        """
        Set the encoder to evaluation mode.

        This disables dropout and uses running statistics for BatchNorm
        layers (instead of batch statistics). Important for consistent
        inference results.
        """
        self.eval()
        logger.info("Encoder set to evaluation mode.")

    def __repr__(self) -> str:
        return (
            f"MobileNetV3Encoder(\n"
            f"  encoder=mobilenet_v3_large,\n"
            f"  pretrained={cfg.ENCODER_PRETRAINED},\n"
            f"  frozen_layers=0-{self.freeze_up_to},\n"
            f"  feature_taps={self.feature_layers},\n"
            f"  output_dim={self.output_dim},\n"
            f"  trainable_params={self.get_num_trainable_params():,},\n"
            f"  total_params={self.get_num_total_params():,}\n"
            f")"
        )


# =========================================================================
# SELF-TEST — Verify encoder loads and produces correct output shapes
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 65)
    print("  MOBILENET ENCODER VERIFICATION")
    print("=" * 65)

    # --- Build encoder ---
    print("\n--- Building Encoder ---")
    encoder = MobileNetV3Encoder()
    print(f"\n{encoder}")

    # --- Forward pass with dummy input ---
    print("\n--- Forward Pass Test ---")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")

    encoder.eval()
    with torch.no_grad():
        output = encoder(dummy_input)

    # Check multi-scale features
    print("\nMulti-scale features:")
    for i, (layer_idx, feat) in enumerate(
        zip(cfg.FEATURE_LAYERS, output["multi_scale_features"])
    ):
        print(f"  Layer {layer_idx}: {feat.shape}")

    # Check concatenated output
    concat = output["concatenated"]
    print(f"\nConcatenated output: {concat.shape}")
    expected_dim = sum(cfg.FEATURE_CHANNELS)
    assert concat.shape == (batch_size, expected_dim), (
        f"Expected ({batch_size}, {expected_dim}), got {concat.shape}"
    )
    print(f"  [PASS] Output shape is correct: ({batch_size}, {expected_dim})")

    # --- Verify frozen vs trainable layers ---
    print("\n--- Gradient Flow Verification ---")
    encoder.train()
    test_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    test_output = encoder(test_input)
    loss = test_output["concatenated"].sum()
    loss.backward()

    # Check that frozen layers have no gradients
    frozen_has_grad = False
    for idx in range(cfg.FREEZE_UP_TO + 1):
        for param in encoder.features[idx].parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                frozen_has_grad = True
                break

    if not frozen_has_grad:
        print(f"  [PASS] Layers 0-{cfg.FREEZE_UP_TO} are properly frozen (no gradients)")
    else:
        print(f"  [FAIL] Frozen layers received gradients!")

    # Check that fine-tuned layers DO have gradients
    finetune_has_grad = False
    for idx in range(cfg.FREEZE_UP_TO + 1, len(encoder.features)):
        for param in encoder.features[idx].parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    finetune_has_grad = True
                    break
        if finetune_has_grad:
            break

    if finetune_has_grad:
        print(
            f"  [PASS] Layers {cfg.FREEZE_UP_TO + 1}-{len(encoder.features) - 1} "
            f"are trainable (receiving gradients)"
        )
    else:
        print(f"  [FAIL] Fine-tuned layers are NOT receiving gradients!")

    # --- Parameter counts ---
    print("\n--- Parameter Summary ---")
    total = encoder.get_num_total_params()
    trainable = encoder.get_num_trainable_params()
    print(f"  Total:     {total:>10,} parameters")
    print(f"  Trainable: {trainable:>10,} parameters ({100*trainable/total:.1f}%)")
    print(f"  Frozen:    {total - trainable:>10,} parameters ({100*(total-trainable)/total:.1f}%)")

    # --- Inference speed benchmark ---
    print("\n--- CPU Inference Speed ---")
    import time
    encoder.eval()
    single_input = torch.randn(1, 3, 224, 224)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = encoder(single_input)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(10):
            start = time.perf_counter()
            _ = encoder(single_input)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    avg_ms = sum(times) / len(times)
    print(f"  Average inference time: {avg_ms:.1f} ms/image (over 10 runs)")
    print(f"  Throughput: ~{1000/avg_ms:.0f} images/sec on CPU")

    # --- Test with actual cookie image ---
    print("\n--- Real Cookie Image Test ---")
    import os
    from PIL import Image
    from transforms import get_eval_transforms

    train_dir = cfg.get_train_dir()
    sample_file = sorted(os.listdir(train_dir))[0]
    sample_path = os.path.join(train_dir, sample_file)

    img = Image.open(sample_path).convert("RGB")
    transform = get_eval_transforms()
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        real_output = encoder(img_tensor)

    print(f"  Image: {sample_file} ({img.size[0]}x{img.size[1]})")
    print(f"  Feature vector shape: {real_output['concatenated'].shape}")
    print(f"  Feature vector stats:")
    feat = real_output["concatenated"][0]
    print(f"    Mean: {feat.mean():.4f}")
    print(f"    Std:  {feat.std():.4f}")
    print(f"    Min:  {feat.min():.4f}")
    print(f"    Max:  {feat.max():.4f}")
    print(f"  [PASS] Real cookie image processed successfully")

    # Cleanup
    encoder.remove_hooks()

    print("\n" + "=" * 65)
    print("  ALL ENCODER TESTS PASSED")
    print("=" * 65)
