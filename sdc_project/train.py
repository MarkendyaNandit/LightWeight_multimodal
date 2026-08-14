"""
train.py — Training Script for the RGB Encoder Module.

Trains the RGBFeatureExtractor (MobileNetV3 + FeatureHead) on normal
Cookie RGB images using self-supervised feature learning.

Training Strategy:
    Since anomaly detection trains ONLY on normal samples (no defect labels),
    we use a self-supervised approach with two loss components:

    1. Consistency Loss (cosine similarity):
       Two randomly augmented views of the SAME image should produce
       nearly identical 256-dim feature vectors.
       L_consist = 1 - cos_sim(f(view_a), f(view_b))

    2. Compactness Loss (center distance):
       ALL normal feature vectors should cluster tightly around a
       learned center. This creates a compact "normal" manifold.
       At test time, anomalous images produce features far from
       this center, enabling detection.
       L_compact = ||f(x) - center||^2

    Total Loss = w_cos * L_consist + w_compact * L_compact

    This mirrors the paper's L_vis (visual-geometric consistency) adapted
    for a single RGB modality. The center plays the role of the "learned
    prototype" in the paper's OCTA module.

Usage:
    python train.py                     # Train with default config
    python train.py --epochs 50         # Override epochs
    python train.py --lr 1e-4           # Override learning rate
    python train.py --resume checkpoint.pth  # Resume from checkpoint

Output:
    outputs/checkpoints/best_model.pth  — Best model (lowest val loss)
    outputs/checkpoints/epoch_*.pth     — Periodic checkpoints
    outputs/logs/training_log.txt       — Training metrics log
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import cfg
from transforms import get_train_transforms, get_eval_transforms
from dataset import CookieTrainDataset
from feature_head import RGBFeatureExtractor

logger = logging.getLogger(__name__)


# =========================================================================
# LOSS FUNCTION
# =========================================================================

class AnomalyFeatureLoss(nn.Module):
    """
    Self-supervised loss for anomaly detection feature learning.

    Combines two complementary objectives:

    1. Consistency Loss: Forces the model to produce stable features
       regardless of augmentation. If two views of the same cookie produce
       very different features, the model hasn't learned robust patterns.

    2. Compactness Loss: Pulls all normal features toward a learnable
       center vector. Creates a tight cluster in 256-D feature space.
       During inference, anomalous images produce features outside this
       cluster, enabling detection via distance thresholding.

    Args:
        feature_dim:       Dimension of feature vectors (256).
        cosine_weight:     Weight for the consistency loss (default: 1.0).
        compactness_weight: Weight for the compactness loss (default: 0.5).
    """

    def __init__(
        self,
        feature_dim: int = cfg.FEATURE_DIM,
        cosine_weight: float = cfg.LOSS_COSINE_WEIGHT,
        compactness_weight: float = cfg.LOSS_COMPACTNESS_WEIGHT,
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.compactness_weight = compactness_weight

        # Learnable center vector (the "normal" prototype).
        # Initialized as zeros and updated during training via gradient descent.
        # After training, this represents the "average normal cookie" in
        # the 256-D feature space.
        self.center = nn.Parameter(
            torch.zeros(1, feature_dim), requires_grad=False
        )

        # Running mean for exponential moving average update of center.
        # We don't backprop through the center — instead we update it
        # with an EMA of the batch means. This is more stable than
        # making the center a learned parameter.
        self.register_buffer(
            "_center_ema", torch.zeros(1, feature_dim)
        )
        self.ema_decay = 0.99  # EMA smoothing factor

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined training loss.

        Args:
            features_a: (B, 256) features from augmented view A.
            features_b: (B, 256) features from augmented view B.

        Returns:
            total_loss: Scalar tensor for backpropagation.
            loss_dict:  Dict with individual loss components for logging.
        """
        # --- Consistency Loss ---
        # Cosine similarity between paired views should be 1.0 (identical).
        # Loss = 1 - mean(cos_sim) -> 0 when views are perfectly aligned.
        cosine_sim = F.cosine_similarity(features_a, features_b, dim=1)
        consistency_loss = (1.0 - cosine_sim).mean()

        # --- Compactness Loss ---
        # Distance from each feature to the running center.
        # Both views should be close to the center.
        center = self.center.detach()  # Don't backprop through center
        dist_a = ((features_a - center) ** 2).sum(dim=1).mean()
        dist_b = ((features_b - center) ** 2).sum(dim=1).mean()
        compactness_loss = (dist_a + dist_b) / 2.0

        # --- Update center with EMA ---
        with torch.no_grad():
            batch_mean = torch.cat([features_a, features_b], dim=0).mean(dim=0, keepdim=True)
            self._center_ema = self.ema_decay * self._center_ema + (1 - self.ema_decay) * batch_mean
            self.center.copy_(self._center_ema)

        # --- Total Loss ---
        total_loss = (
            self.cosine_weight * consistency_loss
            + self.compactness_weight * compactness_loss
        )

        # Build logging dict
        loss_dict = {
            "total_loss": total_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "compactness_loss": compactness_loss.item(),
            "cosine_sim_mean": cosine_sim.mean().item(),
        }

        return total_loss, loss_dict


# =========================================================================
# TRAINING LOOP — ONE EPOCH
# =========================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: AnomalyFeatureLoss,
    device: str,
    epoch: int,
    max_grad_norm: float = cfg.MAX_GRAD_NORM,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Each training step:
        1. Load a batch of normal images (dual-view augmented).
        2. Forward both views through the model.
        3. Compute consistency + compactness loss.
        4. Backward pass with gradient clipping.
        5. Optimizer step.

    Args:
        model:          RGBFeatureExtractor model.
        dataloader:     Training DataLoader (yields view_a, view_b).
        optimizer:      AdamW optimizer.
        loss_fn:        AnomalyFeatureLoss instance.
        device:         "cpu" or "cuda".
        epoch:          Current epoch number (for logging).
        max_grad_norm:  Maximum gradient norm for clipping.

    Returns:
        Dict with averaged training metrics for the epoch.
    """
    model.train()
    loss_fn.train()

    total_loss = 0.0
    total_consistency = 0.0
    total_compactness = 0.0
    total_cosine_sim = 0.0
    num_batches = 0

    epoch_start = time.time()

    for batch_idx, (view_a, view_b) in enumerate(dataloader):
        view_a = view_a.to(device)
        view_b = view_b.to(device)

        # Forward pass: get 256-dim features for both views
        features_a = model(view_a)
        features_b = model(view_b)

        # Compute loss
        loss, loss_dict = loss_fn(features_a, features_b)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

        # Optimizer step
        optimizer.step()

        # Accumulate metrics
        total_loss += loss_dict["total_loss"]
        total_consistency += loss_dict["consistency_loss"]
        total_compactness += loss_dict["compactness_loss"]
        total_cosine_sim += loss_dict["cosine_sim_mean"]
        num_batches += 1

        # Log at intervals
        if (batch_idx + 1) % cfg.LOG_EVERY_N_STEPS == 0 or batch_idx == 0:
            logger.debug(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss_dict['total_loss']:.4f} "
                f"cos_sim={loss_dict['cosine_sim_mean']:.4f}"
            )

    # Average metrics over the epoch
    epoch_time = time.time() - epoch_start
    metrics = {
        "train_loss": total_loss / num_batches,
        "train_consistency_loss": total_consistency / num_batches,
        "train_compactness_loss": total_compactness / num_batches,
        "train_cosine_sim": total_cosine_sim / num_batches,
        "train_time_sec": epoch_time,
    }

    return metrics


# =========================================================================
# VALIDATION LOOP
# =========================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: AnomalyFeatureLoss,
    device: str,
) -> Dict[str, float]:
    """
    Validate the model on held-out normal images.

    Uses the eval transform (no augmentation). Computes compactness
    loss to check if the model is generalizing (not just memorizing
    training images).

    For validation with single-view data, we compute:
    - Compactness loss: distance from features to center
    - Feature statistics: mean, std (should be stable across epochs)

    Args:
        model:      RGBFeatureExtractor in eval mode.
        dataloader: Validation DataLoader (yields (image,) tuples).
        loss_fn:    AnomalyFeatureLoss (uses its center for distance).
        device:     "cpu" or "cuda".

    Returns:
        Dict with validation metrics.
    """
    model.eval()

    all_features = []
    total_compactness = 0.0
    num_batches = 0

    for (images,) in dataloader:
        images = images.to(device)
        features = model(images)

        # Compactness loss (distance to center)
        center = loss_fn.center.detach()
        dist = ((features - center) ** 2).sum(dim=1).mean()
        total_compactness += dist.item()
        num_batches += 1

        all_features.append(features.cpu())

    # Concatenate all validation features
    all_features = torch.cat(all_features, dim=0)

    # Compute pairwise cosine similarity (measures feature consistency)
    sim_matrix = F.cosine_similarity(
        all_features.unsqueeze(0), all_features.unsqueeze(1), dim=2
    )
    # Exclude diagonal (self-similarity = 1.0)
    mask = ~torch.eye(len(all_features), dtype=torch.bool)
    avg_pairwise_sim = sim_matrix[mask].mean().item()

    # Feature space statistics
    feat_mean = all_features.mean(dim=0)
    feat_std = all_features.std(dim=0).mean().item()
    feat_norm = torch.norm(all_features, dim=1).mean().item()

    metrics = {
        "val_loss": total_compactness / max(num_batches, 1),
        "val_pairwise_cosine_sim": avg_pairwise_sim,
        "val_feature_std": feat_std,
        "val_feature_norm": feat_norm,
        "val_num_samples": len(all_features),
    }

    return metrics


# =========================================================================
# EARLY STOPPING
# =========================================================================

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Monitors a metric (default: val_loss) and counts consecutive epochs
    without improvement. If patience is exceeded, training stops.

    Args:
        patience:   Number of epochs to wait for improvement.
        min_delta:  Minimum change to qualify as improvement.
        mode:       "min" (lower is better) or "max" (higher is better).
    """

    def __init__(
        self,
        patience: int = cfg.EARLY_STOPPING_PATIENCE,
        min_delta: float = cfg.EARLY_STOPPING_MIN_DELTA,
        mode: str = cfg.MONITOR_MODE,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: The monitored metric value for this epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


# =========================================================================
# CHECKPOINT MANAGEMENT
# =========================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: AnomalyFeatureLoss,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
) -> None:
    """
    Save a training checkpoint.

    Saves everything needed to resume training or load for inference:
    - Model weights (encoder + head)
    - Optimizer state (momentum, adaptive LR)
    - Scheduler state (current LR)
    - Loss function state (center vector)
    - Training metadata (epoch, metrics)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss_fn_state_dict": loss_fn.state_dict(),
        "metrics": metrics,
        "config": {
            "feature_dim": cfg.FEATURE_DIM,
            "encoder_name": cfg.ENCODER_NAME,
            "freeze_up_to": cfg.FREEZE_UP_TO,
            "image_size": cfg.IMAGE_SIZE,
            "normalize_mean": cfg.NORMALIZE_MEAN,
            "normalize_std": cfg.NORMALIZE_STD,
        },
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    loss_fn: Optional[AnomalyFeatureLoss] = None,
) -> int:
    """
    Load a training checkpoint.

    Args:
        filepath:  Path to the checkpoint file.
        model:     Model to load weights into.
        optimizer: Optimizer to restore state (None to skip).
        scheduler: Scheduler to restore state (None to skip).
        loss_fn:   Loss function to restore center (None to skip).

    Returns:
        The epoch number from the checkpoint.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from epoch {checkpoint['epoch']}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored.")

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state restored.")

    if loss_fn and "loss_fn_state_dict" in checkpoint:
        loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
        logger.info("Loss function state (center) restored.")

    return checkpoint["epoch"]


# =========================================================================
# MAIN TRAINING PIPELINE
# =========================================================================

def setup_logging(log_dir: str, experiment_name: str) -> None:
    """Configure logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}_train.log")

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler (INFO level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))

    # File handler (DEBUG level — captures everything)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to: {log_file}")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cfg.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} (deterministic={cfg.DETERMINISTIC})")


def build_optimizer(model: RGBFeatureExtractor) -> torch.optim.Optimizer:
    """
    Build the optimizer with differential learning rates.

    The encoder (pretrained) gets a 10x lower LR than the head
    (randomly initialized) to prevent catastrophic forgetting.
    """
    param_groups = model.get_param_groups()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.LEARNING_RATE,  # Default LR (overridden by param groups)
        weight_decay=cfg.WEIGHT_DECAY,
        betas=cfg.ADAM_BETAS,
        eps=cfg.ADAM_EPS,
    )
    logger.info(
        f"Optimizer: AdamW (encoder_lr={param_groups[0]['lr']:.1e}, "
        f"head_lr={param_groups[1]['lr']:.1e}, "
        f"weight_decay={cfg.WEIGHT_DECAY})"
    )
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int):
    """Build the learning rate scheduler."""
    if cfg.SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=cfg.SCHEDULER_MIN_LR
        )
    elif cfg.SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.SCHEDULER_STEP_SIZE,
            gamma=cfg.SCHEDULER_GAMMA,
        )
    elif cfg.SCHEDULER == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.MONITOR_MODE,
            patience=cfg.SCHEDULER_PATIENCE,
            factor=cfg.SCHEDULER_GAMMA,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.SCHEDULER}")

    logger.info(f"Scheduler: {cfg.SCHEDULER}")
    return scheduler


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main training pipeline.

    Steps:
        1. Setup (logging, seed, device)
        2. Build datasets and dataloaders
        3. Build model, optimizer, scheduler, loss
        4. Optionally resume from checkpoint
        5. Training loop with validation and early stopping
        6. Save final model and training history
    """
    # --- Parse arguments ---
    if args is None:
        parser = argparse.ArgumentParser(description="Train RGB Encoder")
        parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
        parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
        parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
        parser.add_argument("--resume", type=str, default=None,
                            help="Path to checkpoint to resume from")
        args = parser.parse_args()

    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size

    # --- Setup ---
    setup_logging(cfg.LOG_DIR, cfg.EXPERIMENT_NAME)
    set_seed(cfg.SEED)
    device = cfg.get_device()

    logger.info("=" * 65)
    logger.info("  RGB ENCODER MODULE — TRAINING")
    logger.info("=" * 65)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")

    # --- Datasets ---
    logger.info("\n--- Building Datasets ---")
    train_dataset = CookieTrainDataset(
        image_dir=cfg.get_train_dir(),
        transform=get_train_transforms(),
    )
    val_dataset = CookieTrainDataset(
        image_dir=cfg.get_val_dir(),
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,  # Drop incomplete last batch for stable BatchNorm
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    logger.info(f"Training:   {len(train_dataset)} images, {len(train_loader)} batches")
    logger.info(f"Validation: {len(val_dataset)} images, {len(val_loader)} batches")

    # --- Model ---
    logger.info("\n--- Building Model ---")
    model = RGBFeatureExtractor().to(device)
    logger.info(f"\n{model}")

    # --- Loss, Optimizer, Scheduler ---
    loss_fn = AnomalyFeatureLoss().to(device)

    # Update LR if overridden via command line
    if learning_rate != cfg.LEARNING_RATE:
        cfg.LEARNING_RATE = learning_rate

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, num_epochs)
    early_stopping = EarlyStopping()

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, model, optimizer, scheduler, loss_fn
        )
        logger.info(f"Resuming from epoch {start_epoch}")

    # --- Training History ---
    history = {
        "train_loss": [], "val_loss": [],
        "train_cosine_sim": [], "val_pairwise_cosine_sim": [],
        "learning_rates": [],
    }
    best_val_loss = float("inf")

    # --- Training Loop ---
    logger.info("\n--- Starting Training ---")
    total_start = time.time()

    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )

        # Validate
        val_metrics = {}
        if epoch % cfg.VAL_EVERY_N_EPOCHS == 0:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

        # Update scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if cfg.SCHEDULER == "plateau" and val_metrics:
            scheduler.step(val_metrics["val_loss"])
        else:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_metrics["train_loss"])
        history["train_cosine_sim"].append(train_metrics["train_cosine_sim"])
        history["learning_rates"].append(current_lr)
        if val_metrics:
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_pairwise_cosine_sim"].append(
                val_metrics["val_pairwise_cosine_sim"]
            )

        # Epoch logging
        epoch_time = time.time() - epoch_start
        log_parts = [
            f"Epoch {epoch:3d}/{num_epochs}",
            f"train_loss={train_metrics['train_loss']:.4f}",
            f"cos_sim={train_metrics['train_cosine_sim']:.4f}",
        ]
        if val_metrics:
            log_parts.append(f"val_loss={val_metrics['val_loss']:.4f}")
            log_parts.append(
                f"val_sim={val_metrics['val_pairwise_cosine_sim']:.4f}"
            )
        log_parts.append(f"lr={current_lr:.2e}")
        log_parts.append(f"time={epoch_time:.1f}s")
        logger.info(" | ".join(log_parts))

        # --- Save best model ---
        if val_metrics and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(
                model, optimizer, scheduler, loss_fn,
                epoch, {**train_metrics, **val_metrics}, best_path
            )
            logger.info(f"  >> New best model! val_loss={best_val_loss:.4f}")

        # --- Periodic checkpoint ---
        if epoch % cfg.CKPT_EVERY_N_EPOCHS == 0:
            ckpt_path = os.path.join(
                cfg.CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth"
            )
            save_checkpoint(
                model, optimizer, scheduler, loss_fn,
                epoch, {**train_metrics, **val_metrics}, ckpt_path
            )

        # --- Early stopping ---
        if val_metrics and early_stopping(val_metrics["val_loss"]):
            logger.info(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"No improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    # --- Training Complete ---
    total_time = time.time() - total_start
    logger.info("\n" + "=" * 65)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Total training time: {total_time / 60:.1f} minutes")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth')}")

    # Save training history
    history_path = os.path.join(cfg.LOG_DIR, f"{cfg.EXPERIMENT_NAME}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    # Save final model
    final_path = os.path.join(cfg.CHECKPOINT_DIR, "final_model.pth")
    save_checkpoint(
        model, optimizer, scheduler, loss_fn,
        epoch, {**train_metrics, **val_metrics}, final_path
    )

    # Cleanup hooks
    model.encoder.remove_hooks()


if __name__ == "__main__":
    main()
