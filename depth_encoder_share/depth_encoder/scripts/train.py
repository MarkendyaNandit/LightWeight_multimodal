"""
scripts/train.py

Training entry point for the Depth MobileNetV3 VICReg encoder.

Usage
-----
  # From the depth_encoder/ directory:
  python scripts/train.py

  # Custom config:
  python scripts/train.py --config configs/config.yaml

  # Resume from checkpoint:
  python scripts/train.py --resume checkpoints/epoch_0050.pt

  # Override specific settings via CLI:
  python scripts/train.py --epochs 200 --batch-size 16 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure the depth_encoder/ directory is on sys.path so that
# absolute imports like `from models.depth_encoder import ...` work
# regardless of where the script is called from.
_HERE = Path(__file__).resolve().parent.parent   # depth_encoder/
sys.path.insert(0, str(_HERE))

from data.dataset import get_dataloader
from models.depth_encoder import build_encoder
from models.vicreg import VICRegModel
from training.trainer import Trainer
from utils.logging_utils import print_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Depth MobileNetV3 encoder with VICReg SSL"
    )
    p.add_argument(
        "--config", type=str,
        default=str(_HERE / "configs" / "config.yaml"),
        help="Path to config YAML (default: configs/config.yaml)",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from",
    )
    # Quick overrides (take priority over config.yaml)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--device",     type=str,   default=None,
                   help="Force device: 'cpu', 'cuda', 'cuda:0', etc.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge CLI overrides into the config dict."""
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["optimizer"]["lr"] = args.lr
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args)

    # Reproducibility
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    device = select_device(args.device)
    print(f"\n[Train] Device  : {device}")
    print(f"[Train] Config  : {args.config}")
    print(f"[Train] Seed    : {seed}")

    # ----------------------------------------------------------------
    # Data loaders
    # ----------------------------------------------------------------
    ds_cfg = cfg["dataset"]
    root   = ds_cfg["root"]

    print(f"\n[Train] Building train DataLoader ...")
    train_loader = get_dataloader(
        root=root,
        category=ds_cfg["category"],
        split=ds_cfg["train_split"],
        cfg=cfg,
        ssl_mode=True,
        shuffle=True,
    )
    print(f"  → {len(train_loader.dataset)} samples  |  "
          f"{len(train_loader)} batches/epoch")

    val_loader = None
    try:
        print(f"[Train] Building val DataLoader ...")
        val_loader = get_dataloader(
            root=root,
            category=ds_cfg["category"],
            split=ds_cfg["val_split"],
            cfg=cfg,
            ssl_mode=True,
            shuffle=False,
        )
        print(f"  → {len(val_loader.dataset)} samples  |  "
              f"{len(val_loader)} batches/epoch")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"  [warn] Could not build val loader: {exc}")
        print("  Validation will be skipped.")

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print(f"\n[Train] Building DepthEncoder ...")
    encoder = build_encoder(cfg["model"])
    print(f"  → Backbone : MobileNetV3-Large (pretrained={cfg['model']['pretrained']})")
    print(f"  → Embedding: {cfg['model']['embedding_dim']}-d")
    print(f"  → Feature layers: {cfg['model']['feature_layers']}")

    vicreg_model = VICRegModel(
        encoder=encoder,
        expander_dim=cfg["vicreg"]["expander_dim"],
    )

    n_params_total    = sum(p.numel() for p in vicreg_model.parameters())
    n_params_encoder  = sum(p.numel() for p in encoder.parameters())
    print(f"  → Encoder params : {n_params_encoder:,}")
    print(f"  → Total params   : {n_params_total:,}")

    # ----------------------------------------------------------------
    # Trainer
    # ----------------------------------------------------------------
    trainer = Trainer(
        model=vicreg_model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        resume_from=args.resume,
    )

    trainer.train()

    # Print summary
    metrics_path = Path(cfg["output"]["run_dir"]) / "metrics.jsonl"
    print_summary(metrics_path)


if __name__ == "__main__":
    main()
