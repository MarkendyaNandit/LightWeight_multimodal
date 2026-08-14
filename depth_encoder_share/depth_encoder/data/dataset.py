"""
data/dataset.py

MVTec 3D-AD depth dataset loader.

Loads XYZ TIFF files from the MVTec 3D-AD dataset, extracts depth maps,
and returns two augmented views per sample (for VICReg self-supervised
training) or a single preprocessed tensor (for feature extraction /
validation).

Dataset layout assumed:
  <root>/<category>/<split>/<label>/xyz/<idx>.tiff
  e.g.
  cookie/train/good/xyz/000.tiff
  cookie/validation/good/xyz/000.tiff
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import DepthAugmentation, DepthPreprocessor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MVTec3DDepthDataset(Dataset):
    """
    PyTorch Dataset for MVTec 3D-AD depth maps (XYZ TIFFs).

    In SSL training mode (`ssl_mode=True`) each call to `__getitem__`
    returns a tuple ``(view1, view2, path_str)`` where both views are
    independently-augmented 3-channel float tensors of the same image.

    In inference mode (`ssl_mode=False`) each call returns a single
    preprocessed tensor ``(tensor, path_str)``.

    Parameters
    ----------
    root : str | Path
        Absolute path to the dataset root (parent of `category/`).
    category : str
        Dataset category name, e.g. "cookie".
    split : str
        Split path relative to category, e.g. "train/good" or "validation/good".
    depth_subdir : str
        Sub-directory name containing the `.tiff` depth files (default "xyz").
    preprocessor : DepthPreprocessor
        Converts raw XYZ numpy arrays to 3-channel float tensors.
    augmentation : DepthAugmentation | None
        Applied only in SSL mode.  Pass `None` for inference / validation.
    ssl_mode : bool
        If True, returns two augmented views per sample.
    """

    _TIFF_EXTENSIONS = {".tiff", ".tif"}

    def __init__(
        self,
        root: Union[str, Path],
        category: str,
        split: str,
        depth_subdir: str = "xyz",
        preprocessor: Optional[DepthPreprocessor] = None,
        augmentation: Optional[DepthAugmentation] = None,
        ssl_mode: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.split = split
        self.depth_subdir = depth_subdir
        self.ssl_mode = ssl_mode

        self.preprocessor = preprocessor or DepthPreprocessor()
        self.augmentation = augmentation

        self.file_paths: List[Path] = self._discover_files()
        if len(self.file_paths) == 0:
            raise RuntimeError(
                f"No TIFF files found in {self._depth_dir()}. "
                "Check that the dataset path and split are correct."
            )

    # ------------------------------------------------------------------
    def _depth_dir(self) -> Path:
        return self.root / self.category / self.split / self.depth_subdir

    def _discover_files(self) -> List[Path]:
        depth_dir = self._depth_dir()
        if not depth_dir.exists():
            raise FileNotFoundError(
                f"Depth directory not found: {depth_dir}"
            )
        files = sorted(
            p for p in depth_dir.iterdir()
            if p.suffix.lower() in self._TIFF_EXTENSIONS
        )
        return files

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, str], Tuple[torch.Tensor, str]]:
        path = self.file_paths[idx]
        xyz = self._load_tiff(path)

        # Preprocess: XYZ numpy → 3-channel float tensor (3, H, W)
        tensor = self.preprocessor(xyz)

        if self.ssl_mode:
            if self.augmentation is None:
                # No augmentor provided: return same tensor twice (fallback)
                return tensor, tensor, str(path)
            view1, view2 = self.augmentation(tensor)
            return view1, view2, str(path)
        else:
            return tensor, str(path)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tiff(path: Path) -> np.ndarray:
        """
        Load a TIFF file and return a float32 numpy array.

        MVTec 3D-AD TIFFs are typically (H, W, 3) with channels (X, Y, Z).
        If the file has a different shape it is returned as-is and the
        DepthPreprocessor handles it gracefully.
        """
        try:
            data = tifffile.imread(str(path))
        except Exception as exc:
            raise IOError(f"Failed to read TIFF: {path}") from exc

        if data.dtype != np.float32:
            data = data.astype(np.float32)

        return data

    # ------------------------------------------------------------------
    # Convenience info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MVTec3DDepthDataset("
            f"category={self.category!r}, "
            f"split={self.split!r}, "
            f"n_samples={len(self)}, "
            f"ssl_mode={self.ssl_mode})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    root: Union[str, Path],
    category: str,
    split: str,
    cfg: dict,
    ssl_mode: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build and return a DataLoader from config.

    Parameters
    ----------
    root : str | Path
        Dataset root directory.
    category : str
        Dataset category (e.g. "cookie").
    split : str
        Split relative path (e.g. "train/good").
    cfg : dict
        Full config dict loaded from config.yaml.
    ssl_mode : bool
        SSL training mode (returns two views) vs. inference mode.
    shuffle : bool
        Whether to shuffle the dataset.

    Returns
    -------
    DataLoader
    """
    pre_cfg = cfg["preprocessing"]
    aug_cfg = cfg["augmentation"]
    train_cfg = cfg["training"]

    preprocessor = DepthPreprocessor(
        image_size=pre_cfg["image_size"],
        invalid_fill_method=pre_cfg["invalid_fill_method"],
        normalize_imagenet=pre_cfg["normalize_imagenet"],
    )

    augmentation: Optional[DepthAugmentation] = None
    if ssl_mode:
        augmentation = DepthAugmentation(
            config=aug_cfg,
            image_size=pre_cfg["image_size"],
        )

    dataset = MVTec3DDepthDataset(
        root=root,
        category=category,
        split=split,
        depth_subdir=cfg["dataset"]["depth_subdir"],
        preprocessor=preprocessor,
        augmentation=augmentation,
        ssl_mode=ssl_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=ssl_mode,        # VICReg needs consistent batch sizes
        persistent_workers=(train_cfg["num_workers"] > 0),
    )

    return loader
