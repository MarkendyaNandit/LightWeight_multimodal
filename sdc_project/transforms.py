"""
transforms.py — Image Preprocessing and Augmentation Pipelines.

Defines three transform pipelines for the RGB Encoder module:

1. Training:    Augmentations + normalization. Returns TWO augmented views
                of the same image for consistency-based self-supervised
                learning (augmented view A and view B should produce
                similar 256-dim feature vectors).

2. Validation:  Resize + normalize only. No augmentation. Used to monitor
                training progress without data leakage.

3. Test:        Identical to validation. Used for final evaluation and
                feature extraction.

Why Data Augmentation Matters for This Module:
    We have only 210 normal cookie images for training. Without augmentation,
    the model would memorize the exact pixel patterns instead of learning
    generalizable cookie features. Mild augmentations (flips, small rotation,
    slight color jitter) simulate natural variation without creating images
    that look anomalous.

Why Two Views (Dual-View Training):
    The training loss has a cosine similarity component:
        L_cos = 1 - cos_sim(f(augment_A(x)), f(augment_B(x)))
    This forces the encoder to produce CONSISTENT features regardless of
    minor appearance changes — making the features robust to lighting
    variation, scanner positioning, etc. This is inspired by the paper's
    visual-geometric consistency loss (L_vis), adapted for single-modality.

Integration Note:
    When your teammates integrate the Depth module, they will define their
    own depth-specific transforms. The normalization stats here (ImageNet)
    are specific to the RGB branch and MobileNetV3.
"""

import logging
from typing import Tuple, Optional

import torch
from torchvision import transforms as T

from config import cfg

logger = logging.getLogger(__name__)


class DualViewTransform:
    """
    Applies two independent random augmentations to the same image.

    During training, each image passes through this transform to produce
    two differently-augmented views (view_a, view_b). The encoder processes
    both views, and the consistency loss pulls their feature vectors together.

    This is a standard technique in self-supervised contrastive learning
    (SimCLR, BYOL, etc.), adapted here for anomaly detection where we
    only have normal samples.

    Args:
        base_transform: The augmentation pipeline to apply. Each call
            produces a different random result due to stochastic transforms
            (RandomFlip, RandomRotation, etc.).

    Returns:
        Tuple[Tensor, Tensor]: (view_a, view_b), both of shape (3, H, W).
    """

    def __init__(self, base_transform: T.Compose):
        self.base_transform = base_transform

    def __call__(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the base transform twice to get two different augmented views.

        Args:
            image: PIL Image (raw, un-transformed).

        Returns:
            (view_a, view_b): Two augmented tensor views of the same image.
        """
        view_a = self.base_transform(image)
        view_b = self.base_transform(image)
        return view_a, view_b

    def __repr__(self) -> str:
        return f"DualViewTransform(\n  {self.base_transform}\n)"


def _build_augmentation_pipeline() -> T.Compose:
    """
    Build the stochastic augmentation pipeline for training.

    The pipeline applies these transforms IN ORDER:
        1. Resize to IMAGE_SIZE — ensures consistent input dimensions.
        2. RandomResizedCrop — simulates slight positional variation.
        3. RandomHorizontalFlip — cookies are symmetric horizontally.
        4. RandomVerticalFlip — cookies are symmetric vertically.
        5. RandomRotation — cookies can appear at any orientation.
        6. ColorJitter — simulates minor lighting/scanner variation.
        7. GaussianBlur (probabilistic) — simulates slight defocus.
        8. ToTensor — converts PIL Image [0, 255] to Tensor [0.0, 1.0].
        9. Normalize — applies ImageNet mean/std normalization.

    The augmentations are intentionally MILD for anomaly detection.
    Aggressive augmentations could make normal images look anomalous,
    which would confuse the model during training.

    Returns:
        T.Compose: The composed augmentation pipeline.
    """
    transform_list = []

    # 1. Resize to target size first (MVTec 3D-AD images are 800×800)
    transform_list.append(
        T.Resize(cfg.IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR)
    )

    # 2. Random Resized Crop
    # Scale range (0.85, 1.0) means we crop 85–100% of the image area,
    # then resize back to IMAGE_SIZE. This simulates slight translation
    # and scale variation without losing too much of the cookie.
    transform_list.append(
        T.RandomResizedCrop(
            size=cfg.IMAGE_SIZE,
            scale=cfg.AUG_CROP_SCALE,
            ratio=(0.95, 1.05),  # Keep roughly square (cookies are circular)
            interpolation=T.InterpolationMode.BILINEAR,
        )
    )

    # 3. Random Horizontal Flip
    if cfg.AUG_HFLIP > 0:
        transform_list.append(T.RandomHorizontalFlip(p=cfg.AUG_HFLIP))

    # 4. Random Vertical Flip
    if cfg.AUG_VFLIP > 0:
        transform_list.append(T.RandomVerticalFlip(p=cfg.AUG_VFLIP))

    # 5. Random Rotation
    if cfg.AUG_ROTATION > 0:
        transform_list.append(
            T.RandomRotation(
                degrees=cfg.AUG_ROTATION,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=0,  # Fill rotated corners with black
            )
        )

    # 6. Color Jitter
    brightness, contrast, saturation, hue = cfg.AUG_COLOR_JITTER
    if any(v > 0 for v in cfg.AUG_COLOR_JITTER):
        transform_list.append(
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )

    # 7. Gaussian Blur (probabilistic)
    if cfg.AUG_BLUR_PROB > 0:
        transform_list.append(
            T.RandomApply(
                [T.GaussianBlur(kernel_size=cfg.AUG_BLUR_KERNEL, sigma=(0.1, 2.0))],
                p=cfg.AUG_BLUR_PROB,
            )
        )

    # 8. Convert PIL Image to Tensor (H, W, C) uint8 → (C, H, W) float32
    transform_list.append(T.ToTensor())

    # 9. ImageNet Normalization
    # MobileNetV3 was pretrained with these exact statistics.
    # Applying them ensures the input distribution matches what the
    # pretrained weights expect.
    transform_list.append(
        T.Normalize(mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD)
    )

    return T.Compose(transform_list)


def _build_eval_pipeline() -> T.Compose:
    """
    Build the deterministic evaluation pipeline (validation + test).

    No augmentation — just resize, convert, and normalize.
    This ensures evaluation results are reproducible and not affected
    by random augmentation.

    Pipeline:
        1. Resize to IMAGE_SIZE
        2. ToTensor
        3. Normalize (ImageNet stats)

    Returns:
        T.Compose: The evaluation transform pipeline.
    """
    return T.Compose([
        T.Resize(cfg.IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD),
    ])


# =========================================================================
# PUBLIC API — These functions are called by dataset.py
# =========================================================================

def get_train_transforms() -> DualViewTransform:
    """
    Get the training transform that produces two augmented views.

    Usage in dataset.py:
        transform = get_train_transforms()
        view_a, view_b = transform(pil_image)
        # view_a.shape == view_b.shape == (3, 224, 224)

    Returns:
        DualViewTransform wrapping the augmentation pipeline.
    """
    base = _build_augmentation_pipeline()
    dual = DualViewTransform(base)
    logger.info("Training transforms initialized (dual-view augmentation).")
    logger.debug(f"Augmentation pipeline:\n{base}")
    return dual


def get_eval_transforms() -> T.Compose:
    """
    Get the evaluation transform (for validation and test sets).

    Usage in dataset.py:
        transform = get_eval_transforms()
        tensor = transform(pil_image)
        # tensor.shape == (3, 224, 224)

    Returns:
        T.Compose: Deterministic resize + normalize pipeline.
    """
    pipeline = _build_eval_pipeline()
    logger.info("Evaluation transforms initialized (no augmentation).")
    return pipeline


def get_inference_transforms() -> T.Compose:
    """
    Get transforms for standalone inference / feature extraction.

    Identical to eval transforms, but provided as a separate function
    for clarity in extract_features.py.

    Returns:
        T.Compose: Deterministic resize + normalize pipeline.
    """
    return _build_eval_pipeline()


def denormalize(
    tensor: torch.Tensor,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization purposes.

    When you want to display a processed image (e.g., in a heatmap overlay),
    you need to undo the normalization to recover natural colors.

    Args:
        tensor: Normalized image tensor of shape (C, H, W) or (B, C, H, W).
        mean:   Normalization mean (defaults to ImageNet values from config).
        std:    Normalization std (defaults to ImageNet values from config).

    Returns:
        Denormalized tensor with pixel values in [0, 1].
    """
    if mean is None:
        mean = cfg.NORMALIZE_MEAN
    if std is None:
        std = cfg.NORMALIZE_STD

    # Handle both single image (C, H, W) and batch (B, C, H, W)
    if tensor.dim() == 3:
        mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    elif tensor.dim() == 4:
        mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D.")

    denorm = tensor * std_t + mean_t
    return torch.clamp(denorm, 0.0, 1.0)


# =========================================================================
# SELF-TEST — Run this file directly to verify transforms work
# =========================================================================
if __name__ == "__main__":
    import os
    from PIL import Image

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("  TRANSFORMS VERIFICATION")
    print("=" * 60)

    # Load a sample training image
    train_dir = cfg.get_train_dir()
    sample_files = sorted([
        f for f in os.listdir(train_dir) if f.endswith(cfg.IMAGE_EXT)
    ])

    if not sample_files:
        print(f"ERROR: No images found in {train_dir}")
        exit(1)

    sample_path = os.path.join(train_dir, sample_files[0])
    print(f"\nSample image: {sample_path}")

    image = Image.open(sample_path).convert("RGB")
    print(f"Original size: {image.size} (W x H)")

    # --- Test training transforms (dual-view) ---
    print("\n--- Training Transforms (Dual-View) ---")
    train_tf = get_train_transforms()
    view_a, view_b = train_tf(image)
    print(f"View A shape: {view_a.shape}  dtype: {view_a.dtype}")
    print(f"View B shape: {view_b.shape}  dtype: {view_b.dtype}")
    print(f"View A range: [{view_a.min():.3f}, {view_a.max():.3f}]")
    print(f"View B range: [{view_b.min():.3f}, {view_b.max():.3f}]")

    # Verify the two views are different (random augmentation)
    diff = (view_a - view_b).abs().mean().item()
    print(f"Mean absolute difference between views: {diff:.4f}")
    assert diff > 0.0, "Views should differ due to random augmentation!"
    print("  [PASS] Views are different (augmentation is stochastic)")

    # --- Test eval transforms ---
    print("\n--- Evaluation Transforms ---")
    eval_tf = get_eval_transforms()
    eval_tensor = eval_tf(image)
    print(f"Eval shape: {eval_tensor.shape}  dtype: {eval_tensor.dtype}")
    print(f"Eval range: [{eval_tensor.min():.3f}, {eval_tensor.max():.3f}]")

    # Verify eval is deterministic
    eval_tensor_2 = eval_tf(image)
    eval_diff = (eval_tensor - eval_tensor_2).abs().max().item()
    print(f"Determinism check (max diff between two eval calls): {eval_diff:.6f}")
    assert eval_diff == 0.0, "Eval transform should be deterministic!"
    print("  [PASS] Evaluation transforms are deterministic")

    # --- Test denormalization ---
    print("\n--- Denormalization ---")
    denorm = denormalize(eval_tensor)
    print(f"Denormalized range: [{denorm.min():.3f}, {denorm.max():.3f}]")
    assert 0.0 <= denorm.min() and denorm.max() <= 1.0, "Denorm should be in [0,1]"
    print("  [PASS] Denormalized values in [0, 1]")

    # --- Test batch denormalization ---
    batch = torch.stack([eval_tensor, eval_tensor])
    denorm_batch = denormalize(batch)
    print(f"Batch denorm shape: {denorm_batch.shape}")
    assert denorm_batch.shape == (2, 3, 224, 224)
    print("  [PASS] Batch denormalization works")

    print("\n" + "=" * 60)
    print("  ALL TRANSFORM TESTS PASSED")
    print("=" * 60)
