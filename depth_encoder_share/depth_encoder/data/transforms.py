"""
data/transforms.py

Depth-specific preprocessing and augmentation pipeline for MVTec 3D-AD.

MVTec 3D-AD XYZ TIFFs store float32 (X, Y, Z) organised depth data.
We extract the Z (depth) channel, handle invalid pixels, normalise, and
replicate to 3 channels so the ImageNet-pretrained MobileNetV3 backbone
can consume it directly.
"""

from __future__ import annotations

import random
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Depth Preprocessing (deterministic)
# ---------------------------------------------------------------------------

class DepthPreprocessor:
    """
    Converts a raw XYZ float32 numpy array (H, W, 3) into a 3-channel
    normalised PIL-compatible float tensor (3, 224, 224).

    Steps
    -----
    1. Extract Z channel (index 2 = depth).
    2. Identify invalid pixels: NaN or exactly 0.0 (sensor background).
    3. Fill invalid pixels using the specified strategy (median / min / zero).
    4. Normalise valid pixels to [0, 1] using per-image min/max.
    5. Convert to uint8 PIL image (0-255).
    6. Resize to `image_size × image_size`.
    7. Convert PIL → float tensor (C=1).
    8. Repeat channel to C=3.
    9. Apply ImageNet mean/std normalisation.

    Parameters
    ----------
    image_size : int
        Target spatial resolution (default 224).
    invalid_fill_method : str
        How to fill background/invalid pixels before normalising.
        One of: "median", "min", "zero".
    normalize_imagenet : bool
        Apply ImageNet µ/σ normalisation after channel repeat.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_size: int = 224,
        invalid_fill_method: str = "median",
        normalize_imagenet: bool = True,
    ) -> None:
        self.image_size = image_size
        self.invalid_fill_method = invalid_fill_method
        self.normalize_imagenet = normalize_imagenet

        self._to_tensor = T.ToTensor()
        self._normalize = T.Normalize(
            mean=self._IMAGENET_MEAN,
            std=self._IMAGENET_STD,
        )

    # ------------------------------------------------------------------
    def __call__(self, xyz: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : np.ndarray
            Float32 array of shape (H, W, 3).  The third channel (index 2)
            is the depth (Z).  May contain NaN or 0 for background.

        Returns
        -------
        torch.Tensor
            Shape (3, image_size, image_size), float32.
        """
        depth = self._extract_depth(xyz)
        depth = self._fill_invalid(depth)
        depth = self._normalise(depth)
        pil   = self._to_pil(depth)
        tensor = self._finalize(pil)
        return tensor

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_depth(self, xyz: np.ndarray) -> np.ndarray:
        """Return the Z channel as a (H, W) float32 array."""
        if xyz.ndim == 3 and xyz.shape[2] == 3:
            return xyz[:, :, 2].astype(np.float32)
        elif xyz.ndim == 2:
            # Already a single-channel depth map
            return xyz.astype(np.float32)
        else:
            raise ValueError(
                f"Unexpected XYZ shape {xyz.shape}. Expected (H, W, 3) or (H, W)."
            )

    def _fill_invalid(self, depth: np.ndarray) -> np.ndarray:
        """Replace NaN and zero pixels with a fill value."""
        invalid_mask = ~np.isfinite(depth) | (depth == 0.0)

        if invalid_mask.all():
            # Entire image is invalid — return zeros
            return np.zeros_like(depth)

        valid_values = depth[~invalid_mask]

        if self.invalid_fill_method == "median":
            fill = float(np.median(valid_values))
        elif self.invalid_fill_method == "min":
            fill = float(valid_values.min())
        elif self.invalid_fill_method == "zero":
            fill = 0.0
        else:
            raise ValueError(
                f"Unknown invalid_fill_method: {self.invalid_fill_method!r}. "
                "Choose from 'median', 'min', 'zero'."
            )

        depth = depth.copy()
        depth[invalid_mask] = fill
        return depth

    def _normalise(self, depth: np.ndarray) -> np.ndarray:
        """Per-image min-max normalisation to [0, 1]."""
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min < 1e-8:
            return np.zeros_like(depth)
        return (depth - d_min) / (d_max - d_min)

    def _to_pil(self, depth_01: np.ndarray) -> Image.Image:
        """Convert (H, W) float [0,1] → uint8 PIL image."""
        img_uint8 = (depth_01 * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_uint8, mode="L")

    def _finalize(self, pil: Image.Image) -> torch.Tensor:
        """Resize → ToTensor → repeat 3ch → normalise."""
        pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = self._to_tensor(pil)          # (1, H, W), float32 [0,1]
        tensor = tensor.repeat(3, 1, 1)        # (3, H, W)
        if self.normalize_imagenet:
            tensor = self._normalize(tensor)
        return tensor


# ---------------------------------------------------------------------------
# Depth Augmentation (stochastic — returns two views for VICReg)
# ---------------------------------------------------------------------------

class DepthAugmentation:
    """
    Produces two stochastically-augmented views of a depth PIL image for
    VICReg contrastive pre-training.

    Augmentations are depth-appropriate:
    - No colour jitter (depth has no colour)
    - Random resized crop  ← main spatial diversity
    - Random horizontal flip
    - Optional vertical flip
    - Small random rotation
    - Optional Gaussian blur (simulates sensor noise)
    - Optional random erasing (simulates sensor drop-out / occlusion)

    The view generation is applied *after* the DepthPreprocessor has already
    produced a 3-channel float tensor.  We therefore apply tensor-level ops.

    Parameters
    ----------
    config : dict
        Augmentation config block from config.yaml.
    image_size : int
        Expected spatial resolution (used for erasing scale).
    """

    def __init__(self, config: dict, image_size: int = 224) -> None:
        self.image_size = image_size
        self._build_transforms(config)

    # ------------------------------------------------------------------
    def _build_transforms(self, cfg: dict) -> None:
        ops: list = []

        if cfg.get("random_resized_crop", True):
            scale = tuple(cfg.get("crop_scale", [0.5, 1.0]))
            ops.append(
                T.RandomResizedCrop(
                    self.image_size,
                    scale=scale,
                    ratio=(0.9, 1.1),       # keep roughly square (depth map)
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            )

        if cfg.get("random_horizontal_flip", True):
            ops.append(T.RandomHorizontalFlip(p=0.5))

        if cfg.get("random_vertical_flip", False):
            ops.append(T.RandomVerticalFlip(p=0.5))

        rot = cfg.get("random_rotation_degrees", 15)
        if rot:
            ops.append(T.RandomRotation(degrees=rot, fill=0))

        if cfg.get("gaussian_blur", True):
            kernel = cfg.get("blur_kernel", 3)
            sigma  = tuple(cfg.get("blur_sigma", [0.1, 1.0]))
            ops.append(T.RandomApply([T.GaussianBlur(kernel_size=kernel, sigma=sigma)], p=0.3))

        if cfg.get("random_erasing", True):
            ops.append(
                T.RandomErasing(
                    p=cfg.get("erasing_prob", 0.2),
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value=0,
                )
            )

        self._aug = T.Compose(ops)

    # ------------------------------------------------------------------
    def __call__(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        tensor : torch.Tensor
            Shape (3, H, W) — output of DepthPreprocessor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Two independently-augmented views, each (3, H, W).
        """
        view1 = self._aug(tensor)
        view2 = self._aug(tensor)
        return view1, view2
