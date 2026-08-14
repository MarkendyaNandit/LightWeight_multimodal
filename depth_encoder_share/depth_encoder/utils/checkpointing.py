"""
utils/checkpointing.py

Checkpoint save / load utilities.

Saved checkpoint format (dict):
  {
    "epoch":           int,
    "model_state":     OrderedDict,
    "optimizer_state": dict,
    "best_val_loss":   float,   (optional)
    "config":          dict,    (optional)
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Union[str, Path],
    best_val_loss: Optional[float] = None,
    config: Optional[dict] = None,
) -> None:
    """
    Save a training checkpoint to `path`.

    Parameters
    ----------
    model : nn.Module
        The full model (VICRegModel or DepthEncoder).
    optimizer : Optimizer
        Current optimiser state.
    epoch : int
        Epoch number (1-indexed: the epoch just completed).
    path : str | Path
        File path to write the checkpoint to (.pt).
    best_val_loss : float | None
        Optional best validation loss to save alongside.
    config : dict | None
        Optional config dict to embed in the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    if config is not None:
        payload["config"] = config

    torch.save(payload, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> int:
    """
    Load a checkpoint and restore model (and optionally optimiser) state.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    model : nn.Module
        Model to restore weights into.
    optimizer : Optimizer | None
        If provided, restore optimiser state too.
    device : torch.device | None
        Device to map tensors to (defaults to CPU).
    strict : bool
        Whether to enforce strict key matching in `load_state_dict`.

    Returns
    -------
    int
        The epoch stored in the checkpoint (use as `start_epoch`).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state"], strict=strict)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    epoch = ckpt.get("epoch", 0)
    print(f"[Checkpoint] Loaded from {path}  (epoch {epoch})")
    return epoch


def load_encoder_only(
    path: Union[str, Path],
    model: nn.Module,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> int:
    """
    Convenience wrapper: load only the encoder weights from a VICRegModel
    checkpoint into a standalone DepthEncoder.

    The checkpoint keys under 'encoder.*' are re-mapped so that a plain
    DepthEncoder can load them without needing the full VICRegModel wrapper.

    Parameters
    ----------
    path : str | Path
    model : DepthEncoder
    device : torch.device | None
    strict : bool

    Returns
    -------
    int
        Epoch stored in the checkpoint.
    """
    path = Path(path)
    map_location = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    full_state = ckpt["model_state"]

    # Extract only encoder.* keys and strip the "encoder." prefix
    encoder_state = {
        k[len("encoder."):]: v
        for k, v in full_state.items()
        if k.startswith("encoder.")
    }

    if encoder_state:
        model.load_state_dict(encoder_state, strict=strict)
        print(f"[Checkpoint] Loaded encoder weights from {path}")
    else:
        # Checkpoint is already a bare DepthEncoder (not wrapped)
        model.load_state_dict(full_state, strict=strict)
        print(f"[Checkpoint] Loaded DepthEncoder weights from {path}")

    return ckpt.get("epoch", 0)
