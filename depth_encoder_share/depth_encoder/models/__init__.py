"""
models/__init__.py

Public API for the models package.
"""

from .depth_encoder import DepthEncoder, build_encoder
from .vicreg import VICRegModel

__all__ = [
    "DepthEncoder",
    "build_encoder",
    "VICRegModel",
]
