"""
config.py — Central Configuration for the RGB Encoder Module.

This is the ONLY file you need to edit to change any hyperparameter,
file path, or training setting. Every other module imports from here.

Module Context:
    This RGB Encoder module is Phase 3 of the larger multimodal anomaly
    detection project (arXiv: 2604.22899). It produces a 256-dimensional
    feature vector from Cookie RGB images using MobileNetV3 + transfer
    learning. This feature vector will later be fused with:
        - Depth features (teammate's module)
        - Text features  (teammate's module)
    via the GACM and OCTA modules (other teammates' responsibility).

Hardware Target:
    Intel Core i7 12th Gen (CPU-only). All defaults are tuned for this.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """
    Master configuration for the RGB Encoder module.

    Groups:
        1. Paths            — Dataset location and output directories
        2. Dataset          — Category info, splits, image properties
        3. Model            — Encoder architecture and feature dimensions
        4. Feature Head     — Projection head that produces the 256-dim vector
        5. Training         — Optimizer, scheduler, loss, epochs
        6. Augmentation     — Data augmentation parameters (train only)
        7. Inference        — Feature extraction and evaluation settings
        8. Logging          — Checkpoint frequency, experiment naming
        9. Hardware         — Device, workers, threading
        10. Reproducibility — Random seeds
    """

    # =====================================================================
    # 1. PATHS
    # =====================================================================

    # Root of this module's code. Auto-resolved from this file's location.
    # All relative paths are built from here.
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

    # Root of the MVTec 3D-AD dataset.
    # Structure: DATASET_ROOT/train/good/rgb/000.png
    #            DATASET_ROOT/test/crack/rgb/001.png
    #            DATASET_ROOT/test/crack/gt/001.png
    # To switch to a different category (e.g., bagel), change ONLY this path.
    DATASET_ROOT: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cookie", "cookie"
    )

    # Output directories — created automatically in __post_init__.
    OUTPUT_DIR: str = ""
    CHECKPOINT_DIR: str = ""
    LOG_DIR: str = ""
    VIS_DIR: str = ""
    FEATURE_DIR: str = ""

    def __post_init__(self):
        """Build derived paths and create output directories."""
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "outputs")
        self.CHECKPOINT_DIR = os.path.join(self.OUTPUT_DIR, "checkpoints")
        self.LOG_DIR = os.path.join(self.OUTPUT_DIR, "logs")
        self.VIS_DIR = os.path.join(self.OUTPUT_DIR, "visualizations")
        self.FEATURE_DIR = os.path.join(self.OUTPUT_DIR, "features")

        for d in [self.CHECKPOINT_DIR, self.LOG_DIR, self.VIS_DIR, self.FEATURE_DIR]:
            os.makedirs(d, exist_ok=True)

    # =====================================================================
    # 2. DATASET
    # =====================================================================

    # Product category name. Used in logging and experiment naming.
    CATEGORY: str = "cookie"

    # Defect types present in the test split.
    # Derived from cookie/cookie/class_ids.json:
    #   {"255": "contamination", "254": "crack", "253": "hole", "0": "good"}
    # "combined" contains samples with multiple defect types overlapping.
    DEFECT_TYPES: List[str] = field(
        default_factory=lambda: ["good", "contamination", "crack", "hole", "combined"]
    )

    # Binary labels for image-level anomaly classification.
    #   0 = normal (the "good" folder)
    #   1 = anomalous (any defect folder)
    NORMAL_LABEL: int = 0
    ANOMALY_LABEL: int = 1

    # Image file extension used in the dataset.
    IMAGE_EXT: str = ".png"

    # Input image size (height, width) after resizing.
    # MobileNetV3 was designed for 224×224 (ImageNet standard).
    # All images in MVTec 3D-AD are 800×800 originally; we resize.
    IMAGE_SIZE: Tuple[int, int] = (224, 224)

    # =====================================================================
    # 3. MODEL — MobileNetV3 Encoder
    # =====================================================================

    # Which MobileNetV3 variant to use.
    # "mobilenet_v3_large"  — 5.4M params, higher accuracy, ~75ms/image on CPU
    # "mobilenet_v3_small"  — 2.5M params, faster but less expressive
    # We use "large" for better feature quality (paper uses DINO ViT with 86M).
    ENCODER_NAME: str = "mobilenet_v3_large"

    # Load ImageNet-pretrained weights.
    # WHY TRANSFER LEARNING:
    #   Our Cookie training set has only 210 normal images — far too few to
    #   train a CNN from scratch. ImageNet-pretrained weights provide:
    #     1. Low-level features (edges, textures) that generalize universally.
    #     2. Mid-level features (shapes, patterns) that transfer well.
    #     3. High-level features that we fine-tune for cookie-specific patterns.
    #   This is standard practice in industrial anomaly detection where normal
    #   training data is limited.
    ENCODER_PRETRAINED: bool = True

    # Freeze layers [0, FREEZE_UP_TO] in MobileNetV3's `features` Sequential.
    # MobileNetV3-Large has 17 blocks (indices 0–16):
    #   Layers 0–3:  Early features (edges, colors) — FREEZE these.
    #   Layers 4–8:  Mid features (textures, shapes) — FREEZE these.
    #   Layers 9–16: High features (object parts) — FINE-TUNE these.
    # Freezing the first 8 layers means:
    #   - 60% of the network retains robust ImageNet features.
    #   - 40% adapts to cookie-specific appearance.
    #   - Reduces trainable parameters by ~50%, faster training on CPU.
    FREEZE_UP_TO: int = 8

    # Indices of layers from which to extract multi-scale features.
    # We tap into 3 points along MobileNetV3's backbone to capture:
    #   Layer 7:  low-level textures  (output: 80 channels, 14×14 spatial)
    #   Layer 11: mid-level structure (output: 112 channels, 14×14 spatial)
    #   Layer 13: high-level semantics(output: 160 channels, 7×7 spatial)
    # Multi-scale extraction is critical because surface defects appear at
    # different scales (tiny cracks vs. large contamination blobs).
    FEATURE_LAYERS: List[int] = field(default_factory=lambda: [7, 11, 13])

    # Output channel count at each extraction layer (architecture-dependent).
    # Verified empirically against MobileNetV3-Large:
    #   Layer 7  -> 80ch  (14x14)
    #   Layer 11 -> 112ch (14x14)
    #   Layer 13 -> 160ch (7x7)
    # Do NOT change unless you switch to a different encoder.
    FEATURE_CHANNELS: List[int] = field(default_factory=lambda: [80, 112, 160])

    # =====================================================================
    # 4. FEATURE HEAD — Projection to 256-D
    # =====================================================================

    # Final output dimension of the RGB feature vector.
    # This is the interface contract with your teammates' modules:
    #   Your output:  (batch_size, 256) tensor
    #   Depth module:  expects 256-dim to fuse via GACM
    #   Text module:   expects 256-dim to align via OCTA
    # The paper uses 768 (DINO ViT output dim). We use 256 for CPU efficiency
    # while maintaining enough expressiveness for anomaly discrimination.
    FEATURE_DIM: int = 256

    # Hidden dimension inside the feature head's MLP layers.
    # Provides a capacity bottleneck:
    #   multi-scale features (352 channels) → 512 hidden → 256 output.
    # Trade-off: larger = more capacity but slower and risk overfitting.
    HIDDEN_DIM: int = 512

    # Dropout rate in the feature head.
    # Regularizes against overfitting on 210 training images.
    # Range: 0.0 (no dropout) to 0.5 (aggressive dropout).
    # 0.1 is conservative and suitable for small-dataset fine-tuning.
    DROPOUT: float = 0.1

    # =====================================================================
    # 5. TRAINING
    # =====================================================================

    # --- Optimizer ---
    # AdamW decouples weight decay from the gradient update.
    # This gives better generalization than vanilla Adam, especially
    # when fine-tuning pretrained models on small datasets.
    OPTIMIZER: str = "adamw"

    # Learning rate.
    # For fine-tuning pretrained models, LR should be much smaller than
    # training from scratch (which uses ~0.01–0.1).
    # 3e-4 is the "sweet spot" recommended by the fastai community for
    # fine-tuning CNNs. Range: 1e-4 to 1e-3.
    LEARNING_RATE: float = 3e-4

    # Weight decay (L2 regularization strength).
    # Prevents the fine-tuned layers from drifting too far from pretrained
    # values. 1e-4 is a safe default for AdamW.
    WEIGHT_DECAY: float = 1e-4

    # Adam/AdamW momentum coefficients.
    ADAM_BETAS: Tuple[float, float] = (0.9, 0.999)

    # Adam/AdamW numerical stability epsilon.
    ADAM_EPS: float = 1e-8

    # --- Learning Rate Scheduler ---
    # "cosine" = CosineAnnealingLR: smoothly decays LR from initial to min.
    #   Preferred for fine-tuning because it avoids sudden LR drops.
    # "step"   = StepLR: multiply LR by gamma every step_size epochs.
    # "plateau"= ReduceLROnPlateau: reduce LR when val loss stops improving.
    SCHEDULER: str = "cosine"

    # Minimum LR at the end of cosine schedule (should be ~10–100× < LR).
    SCHEDULER_MIN_LR: float = 1e-6

    # StepLR parameters (only used if SCHEDULER == "step").
    SCHEDULER_STEP_SIZE: int = 20
    SCHEDULER_GAMMA: float = 0.5

    # ReduceLROnPlateau patience (only used if SCHEDULER == "plateau").
    SCHEDULER_PATIENCE: int = 5

    # --- Training Loop ---
    # Total training epochs.
    # With 210 images and batch_size 8 → 27 iterations/epoch.
    # 100 epochs = 2,700 iterations total. On CPU this takes ~15–25 min.
    NUM_EPOCHS: int = 100

    # Batch size for training.
    # On i7 12th Gen (16 GB RAM), batch_size=8 keeps memory under 4 GB.
    # Each 224×224×3 image is ~150 KB; 8 images = ~1.2 MB input tensor.
    BATCH_SIZE: int = 8

    # Batch size for validation/test (no gradients → can be larger).
    EVAL_BATCH_SIZE: int = 16

    # Gradient clipping (max L2 norm).
    # Prevents exploding gradients which can destabilize fine-tuning.
    # 1.0 is a standard safe value. Set to 0.0 to disable.
    MAX_GRAD_NORM: float = 1.0

    # --- Loss ---
    # The training loss combines two complementary objectives:
    #
    # 1. Cosine Similarity Loss (angular alignment):
    #    Ensures feature vectors from augmented views of the SAME normal
    #    image point in the same direction in 256-D space.
    #    Formula: L_cos = 1 - cos_sim(f_view1, f_view2)
    #
    # 2. Compactness Loss (magnitude/clustering):
    #    Pulls all normal feature vectors toward the running mean, creating
    #    a tight cluster. At test time, anomalous images produce features
    #    far from this cluster.
    #    Formula: L_compact = ||f - mean(f)||^2
    #
    # These mirror the paper's L_vis (visual-geometric consistency) adapted
    # for a single RGB modality.
    LOSS_COSINE_WEIGHT: float = 1.0
    LOSS_COMPACTNESS_WEIGHT: float = 0.5

    # --- Early Stopping ---
    # Stop training if validation loss doesn't improve for this many epochs.
    # Prevents overfitting and saves time on CPU.
    EARLY_STOPPING_PATIENCE: int = 15

    # Minimum improvement required to count as "improvement".
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # =====================================================================
    # 6. DATA AUGMENTATION
    # =====================================================================
    # Applied ONLY to training data. Validation and test use clean images.
    #
    # For anomaly detection, augmentations must be MILD — aggressive
    # augmentations (heavy noise, cutout, elastic distortion) can make
    # normal images look anomalous, confusing the model.
    #
    # Cookies are roughly circular and top-down photographed, so:
    #   - Rotation is natural (any angle is a valid cookie orientation)
    #   - Flips are valid (top-down symmetry)
    #   - Color jitter must be mild (lighting variation, not color distortion)

    # Random horizontal flip probability.
    AUG_HFLIP: float = 0.5

    # Random vertical flip probability.
    AUG_VFLIP: float = 0.5

    # Random rotation range (±degrees).
    AUG_ROTATION: int = 15

    # Color jitter (brightness, contrast, saturation, hue).
    # These simulate minor lighting variation in the industrial scanner.
    AUG_COLOR_JITTER: Tuple[float, float, float, float] = (0.1, 0.1, 0.05, 0.02)

    # Random resized crop scale range.
    # (0.85, 1.0) means crop between 85%–100% of the image, then resize to
    # IMAGE_SIZE. Simulates slight positional variation on the scanner.
    AUG_CROP_SCALE: Tuple[float, float] = (0.85, 1.0)

    # Gaussian blur (simulates slight defocus).
    AUG_BLUR_PROB: float = 0.2
    AUG_BLUR_KERNEL: int = 3

    # ImageNet normalization (used because MobileNetV3 was pretrained on it).
    # These MUST match ImageNet statistics — do NOT change these values.
    NORMALIZE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    NORMALIZE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # =====================================================================
    # 7. INFERENCE
    # =====================================================================

    # Gaussian smoothing sigma for anomaly heatmap post-processing.
    # Smooths raw pixel scores for cleaner visualization.
    ANOMALY_MAP_SIGMA: float = 4.0

    # Anomaly score threshold for binary decision (normal vs. anomalous).
    # Tuned on the validation set during evaluation.
    ANOMALY_THRESHOLD: float = 0.5

    # =====================================================================
    # 8. LOGGING & CHECKPOINTING
    # =====================================================================

    # Experiment name — used in log filenames and checkpoint prefixes.
    EXPERIMENT_NAME: str = "rgb_encoder_cookie_v1"

    # Log training metrics every N steps within each epoch.
    LOG_EVERY_N_STEPS: int = 10

    # Run validation every N epochs.
    VAL_EVERY_N_EPOCHS: int = 1

    # Save a checkpoint every N epochs (in addition to the best model).
    CKPT_EVERY_N_EPOCHS: int = 10

    # Keep at most N checkpoints on disk (oldest removed first).
    MAX_CHECKPOINTS: int = 3

    # Metric to track for "best model" saving.
    # "val_loss" → save model with the lowest validation loss.
    MONITOR_METRIC: str = "val_loss"

    # "min" = lower is better (loss), "max" = higher is better (AUROC).
    MONITOR_MODE: str = "min"

    # =====================================================================
    # 9. HARDWARE
    # =====================================================================

    # Device selection.
    # "cpu"  = force CPU (recommended for this project).
    # "auto" = use CUDA if available, otherwise CPU.
    DEVICE: str = "auto"

    # Number of DataLoader workers.
    # Windows multiprocessing with PyTorch can be problematic.
    # 0 = load data in the main process (safest on Windows).
    # 2–4 = parallel loading (use on Linux for speedup).
    NUM_WORKERS: int = 0

    # Pin memory for CPU→GPU transfer (irrelevant on CPU-only systems).
    PIN_MEMORY: bool = False

    # PyTorch intra-op thread count.
    # 0 = use PyTorch default (all available cores).
    # For i7 12th Gen (12 cores / 20 threads), setting to 8–12 is optimal.
    NUM_THREADS: int = 0

    # =====================================================================
    # 10. REPRODUCIBILITY
    # =====================================================================

    # Master seed applied to Python random, NumPy, and PyTorch RNGs.
    SEED: int = 42

    # Deterministic mode. True = reproducible results (slightly slower).
    DETERMINISTIC: bool = True

    # =====================================================================
    # HELPER METHODS
    # =====================================================================

    def get_device(self) -> str:
        """Resolve 'auto' to the actual device string."""
        if self.DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.DEVICE

    def get_train_dir(self) -> str:
        """Path to training RGB images: .../train/good/rgb/"""
        return os.path.join(self.DATASET_ROOT, "train", "good", "rgb")

    def get_val_dir(self) -> str:
        """Path to validation RGB images: .../validation/good/rgb/"""
        return os.path.join(self.DATASET_ROOT, "validation", "good", "rgb")

    def get_test_dir(self) -> str:
        """Path to the test root: .../test/ (contains defect subdirectories)."""
        return os.path.join(self.DATASET_ROOT, "test")

    def get_test_rgb_dir(self, defect_type: str) -> str:
        """Path to test RGB images for a specific defect type."""
        return os.path.join(self.DATASET_ROOT, "test", defect_type, "rgb")

    def get_test_gt_dir(self, defect_type: str) -> str:
        """Path to ground-truth masks for a specific defect type."""
        return os.path.join(self.DATASET_ROOT, "test", defect_type, "gt")

    def summary(self) -> str:
        """Print a human-readable configuration summary."""
        lines = [
            "",
            "=" * 65,
            "  RGB ENCODER MODULE — CONFIGURATION",
            "=" * 65,
            f"  Experiment    : {self.EXPERIMENT_NAME}",
            f"  Category      : {self.CATEGORY}",
            f"  Device        : {self.get_device()}",
            f"  Seed          : {self.SEED}",
            "-" * 65,
            "  DATASET",
            f"    Root        : {self.DATASET_ROOT}",
            f"    Image Size  : {self.IMAGE_SIZE[0]}x{self.IMAGE_SIZE[1]}",
            f"    Train Dir   : {self.get_train_dir()}",
            f"    Val Dir     : {self.get_val_dir()}",
            f"    Test Dir    : {self.get_test_dir()}",
            f"    Defect Types: {self.DEFECT_TYPES}",
            "-" * 65,
            "  MODEL",
            f"    Encoder     : {self.ENCODER_NAME}",
            f"    Pretrained  : {self.ENCODER_PRETRAINED}",
            f"    Frozen 0-{self.FREEZE_UP_TO}  : keeps ImageNet low/mid features",
            f"    Feature Taps: layers {self.FEATURE_LAYERS}  (verified: 80ch, 112ch, 160ch)",
            f"    Tap Channels: {self.FEATURE_CHANNELS}",
            "-" * 65,
            "  FEATURE HEAD",
            f"    Output Dim  : {self.FEATURE_DIM}  (interface to Depth/Text modules)",
            f"    Hidden Dim  : {self.HIDDEN_DIM}",
            f"    Dropout     : {self.DROPOUT}",
            "-" * 65,
            "  TRAINING",
            f"    Optimizer   : {self.OPTIMIZER}",
            f"    LR          : {self.LEARNING_RATE}",
            f"    Weight Decay: {self.WEIGHT_DECAY}",
            f"    Scheduler   : {self.SCHEDULER} (min_lr={self.SCHEDULER_MIN_LR})",
            f"    Epochs      : {self.NUM_EPOCHS}",
            f"    Batch Size  : {self.BATCH_SIZE} (eval: {self.EVAL_BATCH_SIZE})",
            f"    Grad Clip   : {self.MAX_GRAD_NORM}",
            f"    Early Stop  : patience={self.EARLY_STOPPING_PATIENCE}",
            "-" * 65,
            "  LOSS",
            f"    Cosine Wt   : {self.LOSS_COSINE_WEIGHT}",
            f"    Compact Wt  : {self.LOSS_COMPACTNESS_WEIGHT}",
            "-" * 65,
            "  OUTPUT",
            f"    Checkpoints : {self.CHECKPOINT_DIR}",
            f"    Logs        : {self.LOG_DIR}",
            f"    Features    : {self.FEATURE_DIR}",
            "=" * 65,
            "",
        ]
        return "\n".join(lines)


# -------------------------------------------------------------------------
# Module-level singleton — import this in all other files:
#     from config import cfg
# -------------------------------------------------------------------------
cfg = Config()


if __name__ == "__main__":
    # Run directly to verify configuration and paths.
    print(cfg.summary())

    print("--- Path Verification ---")
    paths_to_check = [
        ("Dataset Root", cfg.DATASET_ROOT),
        ("Train RGB Dir", cfg.get_train_dir()),
        ("Val RGB Dir", cfg.get_val_dir()),
        ("Test Root Dir", cfg.get_test_dir()),
        ("Checkpoint Dir", cfg.CHECKPOINT_DIR),
        ("Log Dir", cfg.LOG_DIR),
        ("Feature Dir", cfg.FEATURE_DIR),
    ]
    all_ok = True
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status:7s}] {name:15s} -> {path}")

    if all_ok:
        print("\nAll paths verified successfully.")
    else:
        print("\nWARNING: Some paths are missing. Check DATASET_ROOT.")

    # Count dataset images
    print("\n--- Dataset Image Counts ---")
    train_dir = cfg.get_train_dir()
    val_dir = cfg.get_val_dir()
    if os.path.exists(train_dir):
        train_count = len([f for f in os.listdir(train_dir) if f.endswith(cfg.IMAGE_EXT)])
        print(f"  Train (good): {train_count} images")
    if os.path.exists(val_dir):
        val_count = len([f for f in os.listdir(val_dir) if f.endswith(cfg.IMAGE_EXT)])
        print(f"  Val   (good): {val_count} images")
    for dt in cfg.DEFECT_TYPES:
        rgb_dir = cfg.get_test_rgb_dir(dt)
        if os.path.exists(rgb_dir):
            count = len([f for f in os.listdir(rgb_dir) if f.endswith(cfg.IMAGE_EXT)])
            print(f"  Test  ({dt:15s}): {count} images")
