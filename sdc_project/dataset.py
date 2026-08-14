"""
dataset.py — Custom PyTorch Datasets for the MVTec 3D-AD Cookie RGB Images.

Provides two Dataset classes tailored to the anomaly detection workflow:

1. CookieTrainDataset:
   - Used for TRAINING and VALIDATION splits.
   - Loads ONLY normal ("good") images — this is fundamental to anomaly
     detection: the model never sees defective images during training.
   - Training mode: returns (view_a, view_b) — two augmented views for
     the consistency loss.
   - Validation mode: returns (image_tensor,) — single clean view.

2. CookieTestDataset:
   - Used for TESTING and EVALUATION.
   - Loads ALL defect types (good, crack, contamination, hole, combined).
   - Returns (image_tensor, label, mask, image_path):
       - label: 0 = normal, 1 = anomalous (image-level)
       - mask:  binary defect mask for pixel-level evaluation
       - image_path: for visualization and debugging

MVTec 3D-AD Folder Structure (Cookie):
    cookie/cookie/
    +-- train/good/rgb/          210 normal images
    +-- validation/good/rgb/      22 normal images
    +-- test/
        +-- good/rgb/             28 normal test images
        |   +-- gt/               28 all-zero masks
        +-- crack/rgb/            27 defective images
        |   +-- gt/               27 binary defect masks
        +-- contamination/rgb/    25 defective images
        |   +-- gt/               25 binary defect masks
        +-- hole/rgb/             26 defective images
        |   +-- gt/               26 binary defect masks
        +-- combined/rgb/         25 defective images
            +-- gt/               25 binary defect masks

Integration Note:
    When your teammates build the Depth module, they will create a similar
    dataset class loading from the xyz/ subdirectories. The fusion pipeline
    will pair RGB and Depth samples by matching filenames (e.g., 000.png).
"""

import os
import logging
from typing import Tuple, Optional, List, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import cfg

logger = logging.getLogger(__name__)


class CookieTrainDataset(Dataset):
    """
    Dataset for training and validation on NORMAL cookie images only.

    In unsupervised anomaly detection, the model is trained exclusively
    on defect-free samples. It learns a representation of "what normal
    looks like." At test time, any deviation from this learned normality
    is flagged as an anomaly.

    Two operating modes controlled by the `transform` argument:
        - DualViewTransform → returns (view_a, view_b) for training.
        - T.Compose         → returns (image,) for validation.

    Args:
        image_dir:   Path to the RGB image directory (e.g., .../train/good/rgb/).
        transform:   Transform pipeline from transforms.py.
                     Use get_train_transforms() for training (returns 2 views).
                     Use get_eval_transforms() for validation (returns 1 view).
        return_path: If True, also return the image file path (for debugging).

    Example:
        >>> from transforms import get_train_transforms
        >>> ds = CookieTrainDataset(cfg.get_train_dir(), get_train_transforms())
        >>> view_a, view_b = ds[0]
        >>> view_a.shape  # torch.Size([3, 224, 224])
    """

    def __init__(
        self,
        image_dir: str,
        transform: Callable,
        return_path: bool = False,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.return_path = return_path

        # Discover all image files, sorted for reproducibility.
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}\n"
                f"Check DATASET_ROOT in config.py."
            )

        self.image_paths: List[str] = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(cfg.IMAGE_EXT)
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No {cfg.IMAGE_EXT} images found in {image_dir}"
            )

        logger.info(
            f"CookieTrainDataset initialized: {len(self.image_paths)} images "
            f"from {image_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Load and transform a single normal image.

        Args:
            idx: Index into the image list.

        Returns:
            If transform is DualViewTransform:
                (view_a, view_b) — two (3, H, W) tensors.
            If transform is T.Compose:
                (image,) — single (3, H, W) tensor.
            If return_path is True:
                Appends the file path as the last element.
        """
        img_path = self.image_paths[idx]

        # Load image as RGB PIL Image.
        # .convert("RGB") ensures consistency even if the PNG has an alpha
        # channel or is stored as grayscale.
        image = Image.open(img_path).convert("RGB")

        # Apply the transform pipeline.
        transformed = self.transform(image)

        # DualViewTransform returns a tuple (view_a, view_b).
        # T.Compose returns a single tensor.
        if isinstance(transformed, tuple):
            # Training mode: dual views
            if self.return_path:
                return transformed[0], transformed[1], img_path
            return transformed[0], transformed[1]
        else:
            # Validation mode: single view
            if self.return_path:
                return transformed, img_path
            return (transformed,)


class CookieTestDataset(Dataset):
    """
    Dataset for testing and evaluation across ALL defect types.

    Loads RGB images from every defect subdirectory under the test/ folder
    and assigns binary labels:
        - "good" folder  → label = 0 (normal)
        - All others     → label = 1 (anomalous)

    Also loads ground-truth binary masks from the gt/ subdirectories for
    pixel-level evaluation (P-AUROC, AUPRO).

    Args:
        test_dir:    Root of the test directory (e.g., .../test/).
        transform:   Evaluation transform pipeline (no augmentation).
        defect_types: List of defect type folder names to load.
                      Defaults to cfg.DEFECT_TYPES.
        mask_size:   Size to resize ground-truth masks to (matches IMAGE_SIZE).

    Example:
        >>> from transforms import get_eval_transforms
        >>> ds = CookieTestDataset(cfg.get_test_dir(), get_eval_transforms())
        >>> image, label, mask, path = ds[0]
        >>> image.shape   # torch.Size([3, 224, 224])
        >>> label         # 0 or 1
        >>> mask.shape    # torch.Size([1, 224, 224])
    """

    def __init__(
        self,
        test_dir: str,
        transform: Callable,
        defect_types: Optional[List[str]] = None,
        mask_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.test_dir = test_dir
        self.transform = transform
        self.defect_types = defect_types or cfg.DEFECT_TYPES
        self.mask_size = mask_size or cfg.IMAGE_SIZE

        if not os.path.isdir(test_dir):
            raise FileNotFoundError(
                f"Test directory not found: {test_dir}\n"
                f"Check DATASET_ROOT in config.py."
            )

        # Build the sample list: (image_path, label, mask_path)
        self.samples: List[Tuple[str, int, Optional[str]]] = []
        self._discover_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No test samples found in {test_dir} for "
                f"defect types {self.defect_types}"
            )

        # Count samples per defect type for logging
        label_counts = {"normal": 0, "anomalous": 0}
        defect_counts = {}
        for _, label, _ in self.samples:
            if label == cfg.NORMAL_LABEL:
                label_counts["normal"] += 1
            else:
                label_counts["anomalous"] += 1

        logger.info(
            f"CookieTestDataset initialized: {len(self.samples)} samples "
            f"({label_counts['normal']} normal, "
            f"{label_counts['anomalous']} anomalous) "
            f"from {test_dir}"
        )

    def _discover_samples(self) -> None:
        """
        Walk through test subdirectories and build the sample list.

        For each defect type folder, pairs rgb/*.png with gt/*.png.
        For "good" samples, the mask is all zeros (no defect).
        """
        for defect_type in self.defect_types:
            rgb_dir = os.path.join(self.test_dir, defect_type, "rgb")
            gt_dir = os.path.join(self.test_dir, defect_type, "gt")

            if not os.path.isdir(rgb_dir):
                logger.warning(f"RGB directory not found, skipping: {rgb_dir}")
                continue

            # Determine label: "good" → 0, everything else → 1
            label = cfg.NORMAL_LABEL if defect_type == "good" else cfg.ANOMALY_LABEL

            # Sort for deterministic ordering
            image_files = sorted([
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith(cfg.IMAGE_EXT)
            ])

            for img_file in image_files:
                img_path = os.path.join(rgb_dir, img_file)

                # Find corresponding ground-truth mask
                mask_path = None
                if os.path.isdir(gt_dir):
                    gt_file = os.path.join(gt_dir, img_file)
                    if os.path.isfile(gt_file):
                        mask_path = gt_file

                self.samples.append((img_path, label, mask_path))

            logger.debug(
                f"  {defect_type:15s}: {len(image_files)} images "
                f"(label={label})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, torch.Tensor, str]:
        """
        Load a test sample with its label and ground-truth mask.

        Args:
            idx: Index into the sample list.

        Returns:
            Tuple of:
                image:  (3, H, W) float32 tensor, normalized.
                label:  int, 0 = normal, 1 = anomalous.
                mask:   (1, H, W) float32 tensor, 0.0 or 1.0.
                        For "good" samples, mask is all zeros.
                path:   str, absolute path to the RGB image file.
        """
        img_path, label, mask_path = self.samples[idx]

        # Load and transform RGB image
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        # Load ground-truth mask
        if mask_path is not None and os.path.isfile(mask_path):
            # Ground-truth masks in MVTec 3D-AD are grayscale PNGs.
            # Pixel value > 0 indicates a defective region.
            mask = Image.open(mask_path).convert("L")  # Grayscale
            mask = mask.resize(
                (self.mask_size[1], self.mask_size[0]),  # PIL uses (W, H)
                resample=Image.NEAREST,  # Nearest-neighbor to preserve binary edges
            )
            mask_np = np.array(mask, dtype=np.float32)
            # Binarize: any non-zero pixel is defective
            mask_np = (mask_np > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)
        else:
            # No mask file → create all-zero mask (normal sample)
            mask_tensor = torch.zeros(
                1, self.mask_size[0], self.mask_size[1], dtype=torch.float32
            )

        return image_tensor, label, mask_tensor, img_path


def get_sample_image_size() -> Tuple[int, int]:
    """
    Read the first training image and return its original (W, H) size.

    Useful for verifying that the resize transform is working correctly
    and for documentation purposes.

    Returns:
        (width, height) of the original PNG file.
    """
    train_dir = cfg.get_train_dir()
    files = sorted([
        f for f in os.listdir(train_dir)
        if f.lower().endswith(cfg.IMAGE_EXT)
    ])
    if not files:
        raise RuntimeError(f"No images found in {train_dir}")
    img = Image.open(os.path.join(train_dir, files[0]))
    return img.size  # PIL returns (W, H)


# =========================================================================
# SELF-TEST — Verify dataset loading and output shapes
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from transforms import get_train_transforms, get_eval_transforms

    print("=" * 65)
    print("  DATASET VERIFICATION")
    print("=" * 65)

    # --- Check original image size ---
    orig_size = get_sample_image_size()
    print(f"\nOriginal image size (W x H): {orig_size}")
    print(f"Target image size (H x W):   {cfg.IMAGE_SIZE}")

    # --- Training Dataset ---
    print("\n--- Training Dataset (Dual-View) ---")
    train_ds = CookieTrainDataset(
        image_dir=cfg.get_train_dir(),
        transform=get_train_transforms(),
    )
    print(f"Number of training samples: {len(train_ds)}")

    view_a, view_b = train_ds[0]
    print(f"View A shape: {view_a.shape}  dtype: {view_a.dtype}")
    print(f"View B shape: {view_b.shape}  dtype: {view_b.dtype}")
    assert view_a.shape == (3, 224, 224), f"Unexpected shape: {view_a.shape}"
    assert view_b.shape == (3, 224, 224), f"Unexpected shape: {view_b.shape}"
    print("  [PASS] Training dataset returns correct dual-view shapes")

    # --- Validation Dataset ---
    print("\n--- Validation Dataset (Single-View) ---")
    val_ds = CookieTrainDataset(
        image_dir=cfg.get_val_dir(),
        transform=get_eval_transforms(),
    )
    print(f"Number of validation samples: {len(val_ds)}")

    (val_img,) = val_ds[0]
    print(f"Val image shape: {val_img.shape}  dtype: {val_img.dtype}")
    assert val_img.shape == (3, 224, 224), f"Unexpected shape: {val_img.shape}"
    print("  [PASS] Validation dataset returns correct single-view shape")

    # --- Test Dataset ---
    print("\n--- Test Dataset (All Defect Types) ---")
    test_ds = CookieTestDataset(
        test_dir=cfg.get_test_dir(),
        transform=get_eval_transforms(),
    )
    print(f"Number of test samples: {len(test_ds)}")

    # Check a normal sample
    img, label, mask, path = test_ds[0]
    print(f"\nFirst sample:")
    print(f"  Image shape: {img.shape}  dtype: {img.dtype}")
    print(f"  Label: {label}")
    print(f"  Mask shape: {mask.shape}  dtype: {mask.dtype}")
    print(f"  Mask unique values: {mask.unique().tolist()}")
    print(f"  Path: {os.path.basename(path)}")
    assert img.shape == (3, 224, 224), f"Unexpected shape: {img.shape}"
    assert mask.shape == (1, 224, 224), f"Unexpected mask shape: {mask.shape}"
    print("  [PASS] Test sample has correct shapes")

    # Count labels
    normal_count = sum(1 for _, l, _, _ in test_ds if l == cfg.NORMAL_LABEL)
    anomaly_count = sum(1 for _, l, _, _ in test_ds if l == cfg.ANOMALY_LABEL)
    print(f"\nLabel distribution:")
    print(f"  Normal (label=0):    {normal_count}")
    print(f"  Anomalous (label=1): {anomaly_count}")
    print(f"  Total:               {len(test_ds)}")
    assert normal_count + anomaly_count == len(test_ds)

    # Verify that anomalous samples have non-zero masks
    print("\n--- Mask Verification ---")
    has_defect_mask = False
    for i in range(len(test_ds)):
        _, lbl, msk, pth = test_ds[i]
        if lbl == cfg.ANOMALY_LABEL and msk.sum() > 0:
            has_defect_mask = True
            defect_pixels = (msk > 0).sum().item()
            total_pixels = msk.numel()
            pct = 100.0 * defect_pixels / total_pixels
            print(
                f"  Sample {i}: defect covers {defect_pixels}/{total_pixels} "
                f"pixels ({pct:.1f}%) — {os.path.basename(pth)}"
            )
            break  # Just show one example

    if has_defect_mask:
        print("  [PASS] Anomalous samples have non-zero ground-truth masks")
    else:
        print("  [WARN] No non-zero masks found for anomalous samples")

    # --- Test return_path mode ---
    print("\n--- Return Path Mode ---")
    train_ds_path = CookieTrainDataset(
        image_dir=cfg.get_train_dir(),
        transform=get_eval_transforms(),
        return_path=True,
    )
    result = train_ds_path[0]
    assert len(result) == 2, f"Expected (image, path), got {len(result)} elements"
    img_tensor, img_path = result
    print(f"  Image: {img_tensor.shape}, Path: {os.path.basename(img_path)}")
    print("  [PASS] return_path mode works correctly")

    print("\n" + "=" * 65)
    print("  ALL DATASET TESTS PASSED")
    print("=" * 65)
