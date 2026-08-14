"""
training/trainer.py

Self-supervised VICReg training loop for the DepthEncoder.

Features
--------
- Cosine annealing LR schedule with linear warm-up
- Gradient clipping
- Train + validation loss tracking per epoch
- Best-model checkpointing by validation loss
- Periodic checkpoint saving
- Structured metric logging (JSON lines)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vicreg import VICRegModel
from training.losses import vicreg_loss, VICRegLossOutput
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.logging_utils import MetricsLogger


# ---------------------------------------------------------------------------
# LR schedule helper
# ---------------------------------------------------------------------------

def _cosine_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Linear warm-up followed by cosine annealing.  Returns the new LR."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates VICReg self-supervised training of the DepthEncoder.

    Parameters
    ----------
    model : VICRegModel
        The VICReg-wrapped depth encoder.
    train_loader : DataLoader
        Returns (view1, view2, path) batches.
    val_loader : DataLoader | None
        Returns (view1, view2, path) batches (validation good images).
    cfg : dict
        Full config dict from config.yaml.
    device : torch.device
        Target device.
    resume_from : str | Path | None
        Path to a checkpoint to resume from.
    """

    def __init__(
        self,
        model: VICRegModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: dict,
        device: torch.device,
        resume_from: Optional[str] = None,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        train_cfg  = cfg["training"]
        vicreg_cfg = cfg["vicreg"]
        opt_cfg    = train_cfg["optimizer"]
        sched_cfg  = train_cfg["scheduler"]
        out_cfg    = cfg["output"]

        self.total_epochs    = train_cfg["epochs"]
        self.grad_clip       = train_cfg.get("grad_clip", 1.0)
        self.log_every       = train_cfg.get("log_every", 5)
        self.save_every      = train_cfg.get("save_every", 10)
        self.base_lr         = opt_cfg["lr"]
        self.min_lr          = sched_cfg.get("min_lr", 1e-6)
        self.warmup_epochs   = sched_cfg.get("warmup_epochs", 10)

        self.sim_coeff = vicreg_cfg["sim_coeff"]
        self.std_coeff = vicreg_cfg["std_coeff"]
        self.cov_coeff = vicreg_cfg["cov_coeff"]
        self.eps       = vicreg_cfg.get("eps", 1e-4)

        self.ckpt_dir  = Path(out_cfg["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # Metrics logger
        run_dir = Path(out_cfg["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(run_dir / "metrics.jsonl")

        # State
        self.start_epoch   = 0
        self.best_val_loss = float("inf")

        # Resume from checkpoint
        if resume_from:
            self.start_epoch = load_checkpoint(
                resume_from, self.model, self.optimizer
            )
            print(f"[Trainer] Resumed from epoch {self.start_epoch}: {resume_from}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print(f"  VICReg Depth Encoder Training")
        print(f"  Epochs : {self.total_epochs}  |  Device : {self.device}")
        print(f"  Train  : {len(self.train_loader.dataset)} samples")
        if self.val_loader:
            print(f"  Val    : {len(self.val_loader.dataset)} samples")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.total_epochs):
            lr = _cosine_lr_with_warmup(
                self.optimizer,
                epoch,
                self.total_epochs,
                self.warmup_epochs,
                self.base_lr,
                self.min_lr,
            )

            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch) if self.val_loader else {}
            elapsed       = time.time() - t0

            # Log
            metrics = {
                "epoch":    epoch + 1,
                "lr":       lr,
                "elapsed":  round(elapsed, 2),
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}":   v for k, v in val_metrics.items()},
            }
            self.logger.log(metrics)

            if (epoch + 1) % self.log_every == 0 or epoch == 0:
                self._print_epoch(epoch, lr, train_metrics, val_metrics, elapsed)

            # Save periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                path = self.ckpt_dir / f"epoch_{epoch+1:04d}.pt"
                save_checkpoint(self.model, self.optimizer, epoch + 1, path)

            # Save best model by validation loss (fallback: train loss)
            monitor = val_metrics.get("loss", train_metrics.get("loss", float("inf")))
            if monitor < self.best_val_loss:
                self.best_val_loss = monitor
                best_path = self.ckpt_dir / "best.pt"
                save_checkpoint(self.model, self.optimizer, epoch + 1, best_path)
                if (epoch + 1) % self.log_every == 0 or epoch == 0:
                    print(f"  ✓ New best {'val' if self.val_loader else 'train'} "
                          f"loss = {monitor:.4f}  →  saved to {best_path}")

        # Save final checkpoint
        final_path = self.ckpt_dir / "final.pt"
        save_checkpoint(self.model, self.optimizer, self.total_epochs, final_path)
        print(f"\n[Trainer] Training complete.  Final checkpoint: {final_path}")
        print(f"[Trainer] Best loss: {self.best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Private: single epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        accum = {"loss": 0.0, "invariance": 0.0, "variance": 0.0, "covariance": 0.0}
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1:4d}/{self.total_epochs} [train]",
            leave=False,
            dynamic_ncols=True,
        )

        for view1, view2, _ in pbar:
            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            z1, z2 = self.model(view1, view2)

            loss_out: VICRegLossOutput = vicreg_loss(
                z1, z2,
                sim_coeff=self.sim_coeff,
                std_coeff=self.std_coeff,
                cov_coeff=self.cov_coeff,
                eps=self.eps,
            )

            loss_out.total.backward()

            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            accum["loss"]       += loss_out.total.item()
            accum["invariance"] += loss_out.invariance.item()
            accum["variance"]   += loss_out.variance.item()
            accum["covariance"] += loss_out.covariance.item()
            n_batches += 1

            pbar.set_postfix(loss=f"{loss_out.total.item():.4f}")

        n = max(n_batches, 1)
        return {k: round(v / n, 6) for k, v in accum.items()}

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        accum = {"loss": 0.0, "invariance": 0.0, "variance": 0.0, "covariance": 0.0}
        n_batches = 0

        for batch in self.val_loader:
            # val_loader may return (view1, view2, path) in SSL mode
            # or (tensor, path) in inference mode — handle both
            if len(batch) == 3:
                view1, view2, _ = batch
            else:
                # Single view — duplicate for loss computation
                view1, _ = batch
                view2 = view1

            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)

            z1, z2 = self.model(view1, view2)

            loss_out: VICRegLossOutput = vicreg_loss(
                z1, z2,
                sim_coeff=self.sim_coeff,
                std_coeff=self.std_coeff,
                cov_coeff=self.cov_coeff,
                eps=self.eps,
            )

            accum["loss"]       += loss_out.total.item()
            accum["invariance"] += loss_out.invariance.item()
            accum["variance"]   += loss_out.variance.item()
            accum["covariance"] += loss_out.covariance.item()
            n_batches += 1

        n = max(n_batches, 1)
        return {k: round(v / n, 6) for k, v in accum.items()}

    # ------------------------------------------------------------------

    @staticmethod
    def _print_epoch(
        epoch: int,
        lr: float,
        train: dict,
        val: dict,
        elapsed: float,
    ) -> None:
        val_str = f"  val_loss={val['loss']:.4f}" if val else ""
        print(
            f"Epoch {epoch+1:4d} | "
            f"lr={lr:.2e} | "
            f"train_loss={train['loss']:.4f} "
            f"(inv={train['invariance']:.3f} "
            f"var={train['variance']:.3f} "
            f"cov={train['covariance']:.3f})"
            f"{val_str} | "
            f"{elapsed:.1f}s"
        )
