"""
training/__init__.py

Public API for the training package.
"""

from .losses import vicreg_loss, VICRegLossOutput
from .trainer import Trainer

__all__ = [
    "vicreg_loss",
    "VICRegLossOutput",
    "Trainer",
]
