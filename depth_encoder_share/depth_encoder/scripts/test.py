"""
scripts/test.py

Anomaly detection evaluation for the trained Depth MobileNetV3 encoder.

Strategy
--------
  1. Load pre-extracted train embeddings as the "normal" gallery.
  2. Embed every test image (all defect classes + good).
  3. Compute each test embedding's *minimum cosine distance* to the gallery
     (1 − max cosine similarity).  High distance → anomalous.
  4. Threshold at the best F1 point and report:
       - Per-class anomaly scores
       - Overall AUROC
       - Confusion matrix
       - Optional: embed + re-extract test images on the fly if no cached
         features exist.

Usage
-----
  # Quick run (uses cached features + best.pt):
  python scripts/test.py

  # Re-extract test features before scoring:
  python scripts/test.py --reextract

  # Specific checkpoint:
  python scripts/test.py --checkpoint checkpoints/epoch_0100.pt

  # Change distance threshold:
  python scripts/test.py --threshold 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------
# Make depth_encoder/ importable regardless of CWD
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from data.dataset import MVTec3DDepthDataset
from data.transforms import DepthPreprocessor
from models.depth_encoder import build_encoder
from utils.checkpointing import load_encoder_only


# ==================================================================
# CLI
# ==================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate depth encoder for anomaly detection (k-NN scoring)"
    )
    p.add_argument("--config", type=str,
                   default=str(_HERE / "configs" / "config.yaml"))
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path (default: checkpoints/best.pt)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Anomaly score threshold (default: auto best-F1)")
    p.add_argument("--k", type=int, default=5,
                   help="Number of nearest neighbours (default: 5)")
    p.add_argument("--reextract", action="store_true",
                   help="Re-run inference to (re)create test feature cache")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ==================================================================
# Helpers
# ==================================================================

TEST_DEFECT_CLASSES = ["good", "combined", "contamination", "crack", "hole"]


def cosine_dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance between rows of a (M, D) and b (N, D).
    Returns (M, N) array.  Assumes rows are L2-normalised.
    """
    sim = a @ b.T                         # (M, N)
    return 1.0 - sim                      # (M, N)  distance in [0, 2]


def knn_score(test_emb: np.ndarray, train_emb: np.ndarray, k: int) -> np.ndarray:
    """
    For each test embedding, compute the mean distance to its k nearest
    training neighbours.  Returns (N_test,) anomaly scores.
    """
    dist = cosine_dist_matrix(test_emb, train_emb)   # (N_test, N_train)
    top_k = np.sort(dist, axis=1)[:, :k]             # (N_test, k)
    return top_k.mean(axis=1)                         # (N_test,)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC without sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    auc = np.mean(pos[:, None] > neg[None, :])
    auc += 0.5 * np.mean(pos[:, None] == neg[None, :])
    return float(auc)


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray):
    """Sweep thresholds and return (best_threshold, best_f1)."""
    thresholds = np.unique(scores)
    best_t, best_f1 = thresholds[0], 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def print_sep(char="=", width=60):
    print(char * width)


# ==================================================================
# Feature extraction (on-the-fly for test splits)
# ==================================================================

def extract_split(
    split: str,
    cfg: dict,
    encoder,
    device: torch.device,
    batch_size: int,
    out_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Extract embeddings for one split and cache to out_dir."""
    ds_cfg = cfg["dataset"]
    pre_cfg = cfg["preprocessing"]

    preprocessor = DepthPreprocessor(
        image_size=pre_cfg["image_size"],
        invalid_fill_method=pre_cfg["invalid_fill_method"],
        normalize_imagenet=pre_cfg["normalize_imagenet"],
    )

    try:
        dataset = MVTec3DDepthDataset(
            root=ds_cfg["root"],
            category=ds_cfg["category"],
            split=split,
            depth_subdir=ds_cfg["depth_subdir"],
            preprocessor=preprocessor,
            augmentation=None,
            ssl_mode=False,
        )
    except (FileNotFoundError, RuntimeError):
        return None, None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=False,
    )

    all_emb, all_paths = [], []
    encoder.eval()
    with torch.no_grad():
        for tensors, paths in tqdm(loader, desc=f"  Embedding {split}", leave=False):
            tensors = tensors.to(device)
            emb, _ = encoder(tensors)
            all_emb.append(emb.cpu().numpy())
            all_paths.extend(list(paths))

    emb_np = np.concatenate(all_emb, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", emb_np)
    (out_dir / "paths.txt").write_text("\n".join(all_paths))

    return emb_np, all_paths


# ==================================================================
# Main
# ==================================================================

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_cfg = cfg["output"]
    feat_dir = Path(out_cfg["features_dir"])
    ckpt_path = args.checkpoint or str(Path(out_cfg["checkpoint_dir"]) / "best.pt")

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print_sep()
    print("  Depth Encoder - Anomaly Detection Test")
    print_sep()
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")
    print(f"  k-NN k     : {args.k}")
    print_sep()

    # ------------------------------------------------------------------
    # Load encoder
    # ------------------------------------------------------------------
    encoder = build_encoder(cfg["model"]).to(device)
    if Path(ckpt_path).exists():
        load_encoder_only(ckpt_path, encoder, device=device, strict=False)
        print(f"  [OK] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  [!] Checkpoint not found - using ImageNet-pretrained weights only")
    encoder.eval()

    # ------------------------------------------------------------------
    # Load (or re-extract) train gallery
    # ------------------------------------------------------------------
    train_feat_path = feat_dir / "train_good" / "features.npy"
    if train_feat_path.exists() and not args.reextract:
        train_emb = np.load(train_feat_path)
        print(f"  [OK] Loaded train gallery : {train_emb.shape} from cache")
    else:
        print("  Re-extracting train/good ...")
        train_emb, _ = extract_split(
            cfg["dataset"]["train_split"], cfg, encoder, device,
            args.batch_size, feat_dir / "train_good",
        )
        if train_emb is None:
            print("  [!] train/good split not found. Cannot continue.")
            return

    # ------------------------------------------------------------------
    # Embed each test defect class
    # ------------------------------------------------------------------
    all_scores:  list[float] = []
    all_labels:  list[int]   = []
    class_results: dict[str, dict] = {}

    print()
    print_sep("-")
    print("  Per-class embedding + scoring")
    print_sep("-")

    for defect in TEST_DEFECT_CLASSES:
        split = f"test/{defect}"
        cache_dir = feat_dir / f"test_{defect}"
        feat_file = cache_dir / "features.npy"

        if feat_file.exists() and not args.reextract:
            emb = np.load(feat_file)
            print(f"  {split:<25} -> {emb.shape[0]:3d} samples  [cache]")
        else:
            emb, paths = extract_split(
                split, cfg, encoder, device, args.batch_size, cache_dir
            )
            if emb is None:
                print(f"  {split:<25} -> not found, skipping")
                continue
            print(f"  {split:<25} -> {emb.shape[0]:3d} samples  [extracted]")

        scores = knn_score(emb, train_emb, k=args.k)
        is_anomaly = int(defect != "good")
        labels = [is_anomaly] * len(scores)

        all_scores.extend(scores.tolist())
        all_labels.extend(labels)

        class_results[defect] = {
            "n": len(scores),
            "mean_score": float(scores.mean()),
            "std_score":  float(scores.std()),
            "label": is_anomaly,
        }

    if len(all_scores) == 0:
        print("\n  No test samples found.")
        return

    all_scores_np = np.array(all_scores)
    all_labels_np = np.array(all_labels)

    # ------------------------------------------------------------------
    # Threshold + metrics
    # ------------------------------------------------------------------
    if args.threshold is not None:
        threshold = args.threshold
        preds = (all_scores_np >= threshold).astype(int)
        tp = int(((preds == 1) & (all_labels_np == 1)).sum())
        fp = int(((preds == 1) & (all_labels_np == 0)).sum())
        fn = int(((preds == 0) & (all_labels_np == 1)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
    else:
        threshold, f1 = best_f1_threshold(all_labels_np, all_scores_np)
        preds = (all_scores_np >= threshold).astype(int)
        tp = int(((preds == 1) & (all_labels_np == 1)).sum())
        fp = int(((preds == 1) & (all_labels_np == 0)).sum())
        fn = int(((preds == 0) & (all_labels_np == 1)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)

    tn = int(((preds == 0) & (all_labels_np == 0)).sum())
    auroc = roc_auc(all_labels_np, all_scores_np)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print()
    print_sep()
    print("  Results")
    print_sep()

    print(f"\n  {'Class':<20} {'N':>4}  {'Mean score':>12}  {'Std':>8}  Label")
    print(f"  {'-'*20}  {'-'*4}  {'-'*12}  {'-'*8}  {'-'*8}")
    for defect, r in class_results.items():
        lbl = "anomaly" if r["label"] else "normal "
        print(f"  {defect:<20} {r['n']:>4}  {r['mean_score']:>12.6f}  "
              f"{r['std_score']:>8.6f}  {lbl}")

    print()
    print_sep("-")
    print(f"  Threshold  : {threshold:.6f}  "
          f"{'(auto best-F1)' if args.threshold is None else '(manual)'}")
    print(f"  AUROC      : {auroc:.4f}")
    print(f"  F1 score   : {f1:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print_sep("-")
    print(f"\n  Confusion Matrix (threshold = {threshold:.4f}):")
    print(f"                   Pred Normal   Pred Anomaly")
    print(f"  Actual Normal       {tn:5d}          {fp:5d}")
    print(f"  Actual Anomaly      {fn:5d}          {tp:5d}")
    print()
    print_sep()
    print("  Done.")
    print_sep()


if __name__ == "__main__":
    main()
