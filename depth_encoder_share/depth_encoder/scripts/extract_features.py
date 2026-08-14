"""
scripts/extract_features.py

Feature extraction script for the trained Depth MobileNetV3 encoder.

Loads a trained checkpoint, runs inference on a specified dataset split,
and saves:
  - features.npy     : (N, 256) float32 embedding array
  - paths.txt        : file paths corresponding to each row
  - feature_maps/    : (optional) per-layer intermediate feature maps

Usage
-----
  # Extract train split embeddings using best checkpoint:
  python scripts/extract_features.py

  # Custom split / checkpoint:
  python scripts/extract_features.py --split test/good --checkpoint checkpoints/best.pt

  # Also save intermediate feature maps:
  python scripts/extract_features.py --save-feature-maps

  # All options:
  python scripts/extract_features.py \\
      --config  configs/config.yaml \\
      --split   validation/good \\
      --checkpoint checkpoints/best.pt \\
      --output  features/val_good \\
      --batch-size 32 \\
      --device  cuda
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Ensure depth_encoder/ is on sys.path
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from data.dataset import MVTec3DDepthDataset
from data.transforms import DepthPreprocessor
from models.depth_encoder import DepthEncoder, build_encoder
from utils.checkpointing import load_encoder_only


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract 256-d depth embeddings from a trained checkpoint"
    )
    p.add_argument(
        "--config", type=str,
        default=str(_HERE / "configs" / "config.yaml"),
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file (default: checkpoints/best.pt)",
    )
    p.add_argument(
        "--split", type=str, default=None,
        help="Dataset split relative to category, e.g. 'train/good'. "
             "Defaults to train_split in config.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: features/<split_name>)",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
    )
    p.add_argument(
        "--device", type=str, default=None,
    )
    p.add_argument(
        "--save-feature-maps", action="store_true",
        help="Also save intermediate feature maps per layer.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    out_cfg = cfg["output"]

    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    split = args.split or ds_cfg["train_split"]

    ckpt_path = args.checkpoint or str(
        Path(out_cfg["checkpoint_dir"]) / "best.pt"
    )

    split_name = split.replace("/", "_")
    out_dir = Path(args.output or (Path(out_cfg["features_dir"]) / split_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"\n[Extract] Split      : {split}")
    print(f"[Extract] Checkpoint : {ckpt_path}")
    print(f"[Extract] Output dir : {out_dir}")
    print(f"[Extract] Device     : {device}")

    # ----------------------------------------------------------------
    # Dataset (inference mode — single view, no augmentation)
    # ----------------------------------------------------------------
    pre_cfg = cfg["preprocessing"]
    preprocessor = DepthPreprocessor(
        image_size=pre_cfg["image_size"],
        invalid_fill_method=pre_cfg["invalid_fill_method"],
        normalize_imagenet=pre_cfg["normalize_imagenet"],
    )

    dataset = MVTec3DDepthDataset(
        root=ds_cfg["root"],
        category=ds_cfg["category"],
        split=split,
        depth_subdir=ds_cfg["depth_subdir"],
        preprocessor=preprocessor,
        augmentation=None,
        ssl_mode=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )

    print(f"[Extract] Samples    : {len(dataset)}")

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    encoder: DepthEncoder = build_encoder(cfg["model"])

    if Path(ckpt_path).exists():
        load_encoder_only(ckpt_path, encoder, device=device, strict=False)
    else:
        print(f"[warn] Checkpoint not found at {ckpt_path}. "
              "Running with untrained (ImageNet-pretrained) weights.")

    encoder = encoder.to(device)
    encoder.eval()

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    all_embeddings: list[np.ndarray] = []
    all_paths:      list[str]        = []
    feature_map_accum: dict[int, list[np.ndarray]] = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features", dynamic_ncols=True):
            tensors, paths = batch
            tensors = tensors.to(device, non_blocking=True)

            embeddings, feature_maps = encoder(tensors)   # (B, 256), {layer_idx: tensor}

            all_embeddings.append(embeddings.cpu().numpy())
            all_paths.extend(list(paths))

            if args.save_feature_maps:
                for layer_idx, fmap in feature_maps.items():
                    feature_map_accum[layer_idx].append(fmap.cpu().numpy())

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    embeddings_np = np.concatenate(all_embeddings, axis=0)   # (N, 256)
    assert embeddings_np.shape == (len(dataset), cfg["model"]["embedding_dim"]), \
        f"Unexpected embedding shape: {embeddings_np.shape}"

    emb_path = out_dir / "features.npy"
    np.save(emb_path, embeddings_np)
    print(f"\n[Extract] Saved embeddings : {emb_path}  shape={embeddings_np.shape}")

    paths_file = out_dir / "paths.txt"
    paths_file.write_text("\n".join(all_paths))
    print(f"[Extract] Saved paths      : {paths_file}")

    if args.save_feature_maps:
        fm_dir = out_dir / "feature_maps"
        fm_dir.mkdir(exist_ok=True)
        for layer_idx, fmap_list in feature_map_accum.items():
            fm_np = np.concatenate(fmap_list, axis=0)
            fm_path = fm_dir / f"layer_{layer_idx:02d}.npy"
            np.save(fm_path, fm_np)
            print(f"[Extract] Saved feature maps layer {layer_idx:2d} : "
                  f"{fm_path}  shape={fm_np.shape}")

    # ----------------------------------------------------------------
    # Quick statistics
    # ----------------------------------------------------------------
    print(f"\n[Extract] Embedding statistics:")
    print(f"  shape  : {embeddings_np.shape}")
    print(f"  mean   : {embeddings_np.mean():.6f}")
    print(f"  std    : {embeddings_np.std():.6f}")
    print(f"  min    : {embeddings_np.min():.6f}")
    print(f"  max    : {embeddings_np.max():.6f}")
    norms = np.linalg.norm(embeddings_np, axis=1)
    print(f"  L2 norms: mean={norms.mean():.4f}  std={norms.std():.4f}  "
          f"(should be ≈1.0 — L2 normalised)")

    print(f"\n[Extract] Done.\n")


if __name__ == "__main__":
    main()
