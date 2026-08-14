"""
unified_dataset.py — Unified Anomaly Dataset for All 13 Categories.

Loads and balances data across:
  - 10 MVTec 3D categories (real RGB + XYZ depth)
  - 3 custom categories (RGB only, grayscale depth proxy)

Returns (rgb_tensor, depth_tensor, label, category_idx) for each sample.
"""

import os
import glob
import random
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MVTEC_CATEGORIES = []

CUSTOM_CATEGORIES = ["phone_screen", "car_metal", "pcb"]

ALL_CATEGORIES = CUSTOM_CATEGORIES

CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform():
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.3),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_depth_tiff(path: str) -> Image.Image:
    """Load MVTec 3D XYZ .tiff depth map and convert to a 3-channel image."""
    try:
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Use all 3 XYZ channels — normalize each channel to [0, 255]
            for c in range(3):
                ch = arr[:, :, c]
                valid = np.isfinite(ch)
                if valid.any():
                    cmin, cmax = ch[valid].min(), ch[valid].max()
                    if cmax - cmin > 1e-6:
                        ch = (ch - cmin) / (cmax - cmin) * 255.0
                    else:
                        ch = np.zeros_like(ch)
                    ch[~valid] = 0
                    arr[:, :, c] = ch
                else:
                    arr[:, :, c] = 0
            return Image.fromarray(arr.astype(np.uint8), mode='RGB')
        elif arr.ndim == 2:
            # Single-channel depth — replicate
            valid = np.isfinite(arr)
            if valid.any():
                dmin, dmax = arr[valid].min(), arr[valid].max()
                if dmax - dmin > 1e-6:
                    arr = (arr - dmin) / (dmax - dmin) * 255.0
                else:
                    arr = np.zeros_like(arr)
            arr[~valid] = 0
            gray = arr.astype(np.uint8)
            return Image.merge('RGB', [Image.fromarray(gray)] * 3)
        else:
            # Fallback
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    except Exception:
        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


def rgb_to_grayscale_depth(rgb_image: Image.Image) -> Image.Image:
    """Convert RGB image to grayscale and replicate to 3 channels (depth proxy)."""
    gray = rgb_image.convert("L")
    return Image.merge("RGB", [gray, gray, gray])


# ── File Discovery ───────────────────────────────────────────────────

def discover_mvtec_files(category: str, split: str = "train") -> List[dict]:
    """
    Discover MVTec 3D files for a given category and split.
    
    split: 'train' -> only good (label=0)
           'validation' -> only good (label=0) 
           'test' -> good (0) + all defect types (1)
    """
    cat_dir = os.path.join(BASE_DIR, "mvtec_3d_anomaly_detection", category)
    samples = []
    
    if split in ("train", "validation"):
        rgb_dir = os.path.join(cat_dir, split, "good", "rgb")
        xyz_dir = os.path.join(cat_dir, split, "good", "xyz")
        if os.path.isdir(rgb_dir):
            for f in sorted(os.listdir(rgb_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    rgb_path = os.path.join(rgb_dir, f)
                    # Corresponding depth
                    depth_name = os.path.splitext(f)[0] + ".tiff"
                    depth_path = os.path.join(xyz_dir, depth_name)
                    if not os.path.isfile(depth_path):
                        depth_path = None
                    samples.append({
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "label": 0,
                        "category": category,
                        "has_real_depth": depth_path is not None
                    })
    elif split == "test":
        split_dir = os.path.join(cat_dir, "test")
        if os.path.isdir(split_dir):
            for defect_type in sorted(os.listdir(split_dir)):
                defect_dir = os.path.join(split_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                rgb_dir = os.path.join(defect_dir, "rgb")
                xyz_dir = os.path.join(defect_dir, "xyz")
                label = 0 if defect_type == "good" else 1
                if os.path.isdir(rgb_dir):
                    for f in sorted(os.listdir(rgb_dir)):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                            rgb_path = os.path.join(rgb_dir, f)
                            depth_name = os.path.splitext(f)[0] + ".tiff"
                            depth_path = os.path.join(xyz_dir, depth_name)
                            if not os.path.isfile(depth_path):
                                depth_path = None
                            samples.append({
                                "rgb_path": rgb_path,
                                "depth_path": depth_path,
                                "label": label,
                                "category": category,
                                "has_real_depth": depth_path is not None
                            })
    return samples


def discover_phone_screen_files() -> List[dict]:
    """Discover phone_screen files from both original archive and MSD-US datasets."""
    samples = []
    
    # 1. Original archive
    good_dir = os.path.join(BASE_DIR, "archive", "good")
    defect_dir = os.path.join(BASE_DIR, "archive", "scratch")
    
    if os.path.isdir(good_dir):
        for f in sorted(os.listdir(good_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append({
                    "rgb_path": os.path.join(good_dir, f),
                    "depth_path": None,
                    "label": 0,
                    "category": "phone_screen",
                    "has_real_depth": False
                })
    if os.path.isdir(defect_dir):
        for f in sorted(os.listdir(defect_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append({
                    "rgb_path": os.path.join(defect_dir, f),
                    "depth_path": None,
                    "label": 1,
                    "category": "phone_screen",
                    "has_real_depth": False
                })

    # 2. MSD-US Dataset
    msd_good = os.path.join(BASE_DIR, "MSD-US", "train", "good")
    msd_defects = [
        os.path.join(BASE_DIR, "MSD-US", "test", "oil"),
        os.path.join(BASE_DIR, "MSD-US", "test", "scratch"),
        os.path.join(BASE_DIR, "MSD-US", "test", "stain")
    ]

    if os.path.isdir(msd_good):
        for f in sorted(os.listdir(msd_good)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                samples.append({
                    "rgb_path": os.path.join(msd_good, f),
                    "depth_path": None,
                    "label": 0,
                    "category": "phone_screen",
                    "has_real_depth": False
                })
                
    for d_dir in msd_defects:
        if os.path.isdir(d_dir):
            for f in sorted(os.listdir(d_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    samples.append({
                        "rgb_path": os.path.join(d_dir, f),
                        "depth_path": None,
                        "label": 1,
                        "category": "phone_screen",
                        "has_real_depth": False
                    })

    return samples


def discover_pcb_files() -> List[dict]:
    """Discover PCB files. Normal=*_temp.jpg, Defect=*_test.jpg."""
    samples = []
    pcb_root = os.path.join(BASE_DIR, "DeepPCB-master", "DeepPCB-master", "PCBData")
    
    # Normal (template/reference)
    temp_files = glob.glob(os.path.join(pcb_root, "**", "*_temp.jpg"), recursive=True)
    for f in sorted(temp_files):
        samples.append({
            "rgb_path": f,
            "depth_path": None,
            "label": 0,
            "category": "pcb",
            "has_real_depth": False
        })
    
    # Defect (test images with defects)
    test_files = glob.glob(os.path.join(pcb_root, "**", "*_test.jpg"), recursive=True)
    for f in sorted(test_files):
        samples.append({
            "rgb_path": f,
            "depth_path": None,
            "label": 1,
            "category": "pcb",
            "has_real_depth": False
        })
    return samples


def discover_car_metal_files() -> List[dict]:
    """Discover car_metal (NEU-DET) files. Normal=crazing, Defect=5 other classes."""
    samples = []
    images_dir = os.path.join(BASE_DIR, "archive (1)", "NEU-DET", "train", "images")
    
    normal_classes = ["crazing"]
    defect_classes = ["inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    
    for cls in normal_classes:
        cls_dir = os.path.join(images_dir, cls)
        if os.path.isdir(cls_dir):
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    samples.append({
                        "rgb_path": os.path.join(cls_dir, f),
                        "depth_path": None,
                        "label": 0,
                        "category": "car_metal",
                        "has_real_depth": False
                    })
    
    for cls in defect_classes:
        cls_dir = os.path.join(images_dir, cls)
        if os.path.isdir(cls_dir):
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    samples.append({
                        "rgb_path": os.path.join(cls_dir, f),
                        "depth_path": None,
                        "label": 1,
                        "category": "car_metal",
                        "has_real_depth": False
                    })
    return samples


# ── Unified Dataset ──────────────────────────────────────────────────

class UnifiedAnomalyDataset(Dataset):
    """
    Unified dataset loading all 13 categories.
    
    Args:
        split: 'train', 'val', or 'test'
        categories: List of category names to load. None = all 13.
        transform: Transform for RGB images.
        depth_transform: Transform for depth images. If None, uses same as RGB.
        normal_only: If True, only load normal (label=0) samples.
        val_ratio: Fraction of custom-category data used for validation.
    
    Returns per __getitem__:
        (rgb_tensor, depth_tensor, label, category_idx)
    """
    
    def __init__(
        self,
        split: str = "train",
        categories: Optional[List[str]] = None,
        transform=None,
        depth_transform=None,
        normal_only: bool = False,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.categories = categories or ALL_CATEGORIES
        self.transform = transform or (get_train_transform() if split == "train" else get_eval_transform())
        self.depth_transform = depth_transform or get_eval_transform()
        self.normal_only = normal_only
        self.val_ratio = val_ratio
        self.seed = seed
        
        self.samples: List[dict] = []
        self._discover_all_files()
        
        logger.info(f"UnifiedAnomalyDataset [{split}]: {len(self.samples)} samples across {len(self.categories)} categories")
    
    def _discover_all_files(self):
        rng = random.Random(self.seed)
        
        for cat in self.categories:
            if cat in MVTEC_CATEGORIES:
                if self.split == "train":
                    samples = discover_mvtec_files(cat, "train")
                elif self.split == "val":
                    samples = discover_mvtec_files(cat, "validation")
                elif self.split == "test":
                    samples = discover_mvtec_files(cat, "test")
                else:
                    samples = []
                    
            elif cat == "phone_screen":
                all_samples = discover_phone_screen_files()
                samples = self._split_custom(all_samples, rng)
                
            elif cat == "pcb":
                all_samples = discover_pcb_files()
                samples = self._split_custom(all_samples, rng)
                
            elif cat == "car_metal":
                all_samples = discover_car_metal_files()
                samples = self._split_custom(all_samples, rng)
            else:
                continue
            
            if self.normal_only:
                samples = [s for s in samples if s["label"] == 0]
            
            self.samples.extend(samples)
    
    def _split_custom(self, all_samples: List[dict], rng: random.Random) -> List[dict]:
        """Split custom dataset into train/val/test (60/20/20)."""
        normal = [s for s in all_samples if s["label"] == 0]
        defect = [s for s in all_samples if s["label"] == 1]
        
        rng_copy = random.Random(self.seed)
        rng_copy.shuffle(normal)
        rng_copy.shuffle(defect)
        
        # 60% train, 20% val, 20% test
        n_val_norm = max(1, int(len(normal) * self.val_ratio))
        n_test_norm = max(1, int(len(normal) * self.val_ratio))
        n_val_def = max(1, int(len(defect) * self.val_ratio))
        n_test_def = max(1, int(len(defect) * self.val_ratio))
        
        if self.split == "train":
            return normal[n_val_norm + n_test_norm:] + defect[n_val_def + n_test_def:]
        elif self.split == "val":
            return normal[:n_val_norm] + defect[:n_val_def]
        elif self.split == "test":
            return normal[n_val_norm:n_val_norm + n_test_norm] + defect[n_val_def:n_val_def + n_test_def]
        return []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sample = self.samples[idx]
        
        # Load RGB
        try:
            rgb_img = Image.open(sample["rgb_path"]).convert("RGB")
        except Exception:
            rgb_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Load Depth
        if sample["has_real_depth"] and sample["depth_path"] is not None:
            depth_img = load_depth_tiff(sample["depth_path"])
        else:
            depth_img = rgb_to_grayscale_depth(rgb_img)
        
        # Apply transforms
        rgb_tensor = self.transform(rgb_img)
        depth_tensor = self.depth_transform(depth_img)
        
        label = sample["label"]
        cat_idx = CATEGORY_TO_IDX[sample["category"]]
        
        return rgb_tensor, depth_tensor, label, cat_idx
    
    def get_category_name(self, cat_idx: int) -> str:
        return ALL_CATEGORIES[cat_idx]
    
    def get_sample_weights(self) -> torch.Tensor:
        """Compute per-sample weights for balanced sampling across categories and labels."""
        # Count samples per (category, label) group
        group_counts: Dict[Tuple[str, int], int] = {}
        for s in self.samples:
            key = (s["category"], s["label"])
            group_counts[key] = group_counts.get(key, 0) + 1
        
        total = len(self.samples)
        n_groups = len(group_counts)
        
        weights = []
        for s in self.samples:
            key = (s["category"], s["label"])
            # Weight inversely proportional to group size
            w = total / (n_groups * group_counts[key])
            weights.append(w)
        
        return torch.tensor(weights, dtype=torch.float64)


def create_dataloaders(
    split: str = "train",
    batch_size: int = 16,
    categories: Optional[List[str]] = None,
    normal_only: bool = False,
    balanced: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, Dataset]:
    """Create a DataLoader for the given split with optional balanced sampling."""
    ds = UnifiedAnomalyDataset(split=split, categories=categories, normal_only=normal_only)
    
    sampler = None
    shuffle = (split == "train") and not balanced
    
    if split == "train" and balanced and len(ds) > 0:
        weights = ds.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        shuffle = False
    
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=(split == "train")
    )
    return loader, ds


# ── Self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("=" * 65)
    print("  UNIFIED ANOMALY DATASET VERIFICATION")
    print("=" * 65)
    
    for split in ["train", "val", "test"]:
        ds = UnifiedAnomalyDataset(split=split)
        cat_counts = {}
        label_counts = {0: 0, 1: 0}
        for s in ds.samples:
            cat_counts[s["category"]] = cat_counts.get(s["category"], 0) + 1
            label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
        
        print(f"\n--- {split.upper()} split: {len(ds)} total samples ---")
        print(f"  Normal: {label_counts[0]}, Defect: {label_counts[1]}")
        for cat in ALL_CATEGORIES:
            print(f"  {cat:15s}: {cat_counts.get(cat, 0)}")
    
    # Test loading one batch
    print("\n--- Loading one training batch ---")
    loader, _ = create_dataloaders("train", batch_size=4)
    rgb, depth, labels, cat_idxs = next(iter(loader))
    print(f"  RGB shape:   {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Labels:      {labels.tolist()}")
    print(f"  Categories:  {cat_idxs.tolist()}")
    print("  [PASS] Dataset loading works!")
