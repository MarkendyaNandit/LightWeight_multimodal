"""
utils.py — Utility Functions for the RGB Encoder Module.

Provides reusable helper functions used across multiple scripts:

    - set_seed()          : Reproducibility across Python, NumPy, PyTorch.
    - get_device()        : Resolve device string ("auto" -> "cpu" or "cuda").
    - count_parameters()  : Count total and trainable parameters in a model.
    - format_time()       : Convert seconds to human-readable string.
    - inspect_checkpoint(): Print contents of a saved checkpoint file.
    - visualize_features(): t-SNE visualization of extracted feature vectors.

This file centralizes utility logic so that train.py, test.py, validate.py,
and extract_features.py remain focused on their primary responsibilities.

Integration Note:
    When the full multimodal system is assembled, these utilities can be
    extended (e.g., adding depth feature visualization) without modifying
    the existing RGB-specific code.
"""

import os
import sys
import time
import random
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import cfg

logger = logging.getLogger(__name__)


# =========================================================================
# REPRODUCIBILITY
# =========================================================================

def set_seed(seed: int = cfg.SEED) -> None:
    """
    Set random seeds for full reproducibility.

    Seeds Python's random module, NumPy, and all PyTorch RNGs.
    Optionally enables deterministic mode for CuDNN.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cfg.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} (deterministic={cfg.DETERMINISTIC})")


# =========================================================================
# DEVICE
# =========================================================================

def get_device() -> torch.device:
    """
    Resolve the compute device from config.

    Returns:
        torch.device: Resolved device object.
    """
    device_str = cfg.get_device()
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    return device


# =========================================================================
# MODEL INSPECTION
# =========================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters broken down by trainability.

    Args:
        model: Any PyTorch module.

    Returns:
        Dict with keys: "total", "trainable", "frozen".
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def print_model_summary(model: nn.Module, name: str = "Model") -> None:
    """
    Print a formatted summary of model parameter counts.

    Args:
        model: PyTorch module.
        name:  Display name for the model.
    """
    counts = count_parameters(model)
    print(f"\n{'=' * 50}")
    print(f"  {name} — Parameter Summary")
    print(f"{'=' * 50}")
    print(f"  Total:     {counts['total']:>12,}")
    print(f"  Trainable: {counts['trainable']:>12,}  "
          f"({100 * counts['trainable'] / max(counts['total'], 1):.1f}%)")
    print(f"  Frozen:    {counts['frozen']:>12,}  "
          f"({100 * counts['frozen'] / max(counts['total'], 1):.1f}%)")
    print(f"  Size:      ~{counts['total'] * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"{'=' * 50}\n")


# =========================================================================
# CHECKPOINT INSPECTION
# =========================================================================

def inspect_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Load and display the contents of a checkpoint file.

    Useful for debugging — shows what epoch the checkpoint was saved at,
    which metrics were recorded, and the config it was trained with.

    Args:
        filepath: Path to a .pth checkpoint file.

    Returns:
        Dict with checkpoint metadata (does NOT return full state dicts).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

    info = {
        "filepath": filepath,
        "file_size_mb": os.path.getsize(filepath) / (1024 * 1024),
        "epoch": checkpoint.get("epoch", "N/A"),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "keys": list(checkpoint.keys()),
    }

    print(f"\n{'=' * 55}")
    print(f"  Checkpoint: {os.path.basename(filepath)}")
    print(f"{'=' * 55}")
    print(f"  File size:  {info['file_size_mb']:.1f} MB")
    print(f"  Epoch:      {info['epoch']}")
    print(f"  Keys:       {info['keys']}")

    if info["metrics"]:
        print(f"\n  Metrics:")
        for key, val in info["metrics"].items():
            if isinstance(val, float):
                print(f"    {key:30s}: {val:.6f}")
            else:
                print(f"    {key:30s}: {val}")

    if info["config"]:
        print(f"\n  Config:")
        for key, val in info["config"].items():
            print(f"    {key:20s}: {val}")

    print(f"{'=' * 55}\n")
    return info


# =========================================================================
# FORMATTING
# =========================================================================

def format_time(seconds: float) -> str:
    """
    Convert seconds to a human-readable time string.

    Examples:
        format_time(65)    -> "1m 5s"
        format_time(3661)  -> "1h 1m 1s"
        format_time(0.5)   -> "0.5s"

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def format_number(num: int) -> str:
    """Format a large number with commas: 1234567 -> '1,234,567'."""
    return f"{num:,}"


# =========================================================================
# FEATURE VISUALIZATION (t-SNE)
# =========================================================================

def visualize_features_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Feature Space Visualization (t-SNE)",
    label_names: Optional[Dict[int, str]] = None,
) -> None:
    """
    Visualize 256-dim feature vectors in 2D using t-SNE.

    Plots normal vs. anomalous features to verify that the encoder
    produces separable representations. Good separation = the model
    learned meaningful features.

    Args:
        features:    (N, 256) array of feature vectors.
        labels:      (N,) array of labels (0=normal, 1=anomalous).
        save_path:   If provided, save the plot to this path.
        title:       Plot title.
        label_names: Optional dict mapping label ints to display names.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib or sklearn not available. "
            "Skipping t-SNE visualization."
        )
        return

    if label_names is None:
        label_names = {0: "Normal", 1: "Anomalous"}

    logger.info(f"Running t-SNE on {len(features)} feature vectors...")
    perplexity = min(30, len(features) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=cfg.SEED,
        n_iter=1000,
    )
    embeddings = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(labels.tolist()))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names.get(label, f"Label {label}")
        color = colors[i % len(colors)]
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=color,
            label=f"{name} (n={mask.sum()})",
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"t-SNE plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# =========================================================================
# SELF-TEST
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 55)
    print("  UTILS VERIFICATION")
    print("=" * 55)

    # --- Test set_seed ---
    print("\n--- Seed Test ---")
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.equal(a, b), "Seeds not working!"
    print("  [PASS] set_seed produces identical random values")

    # --- Test get_device ---
    print("\n--- Device Test ---")
    device = get_device()
    print(f"  Device: {device}")
    print("  [PASS] get_device works")

    # --- Test format_time ---
    print("\n--- Format Time ---")
    assert format_time(0.5) == "0.5s"
    assert format_time(65) == "1m 5s"
    assert format_time(3661) == "1h 1m 1s"
    print("  [PASS] format_time works")

    # --- Test count_parameters ---
    print("\n--- Count Parameters ---")
    model = nn.Linear(100, 50)
    counts = count_parameters(model)
    assert counts["total"] == 100 * 50 + 50  # weights + bias
    assert counts["trainable"] == counts["total"]
    assert counts["frozen"] == 0
    print(f"  Linear(100, 50): {counts['total']} params")
    print("  [PASS] count_parameters works")

    # --- Test checkpoint inspection ---
    print("\n--- Checkpoint Inspection ---")
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    if os.path.isfile(ckpt_path):
        info = inspect_checkpoint(ckpt_path)
        print("  [PASS] Checkpoint inspection works")
    else:
        print("  [SKIP] No checkpoint found (run train.py first)")

    print("\n" + "=" * 55)
    print("  ALL UTILS TESTS PASSED")
    print("=" * 55)
