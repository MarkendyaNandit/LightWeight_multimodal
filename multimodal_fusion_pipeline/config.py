"""
config.py — Configuration for Multimodal Fusion and GACM Pipeline.

Configures paths, model parameters, and training settings for Phase 6 (Fusion),
Phase 7 (GACM), Phase 9 (Training), and Phase 10 (Inference & Text Comparison).
"""

import os
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class FusionConfig:
    # Root directory
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR: str = os.path.dirname(PROJECT_ROOT)
    
    # Feature Input Paths
    RGB_FEATURE_DIR: str = os.path.join(BASE_DIR, "sdc_project", "outputs", "features")
    DEPTH_FEATURE_DIR: str = os.path.join(BASE_DIR, "depth_encoder_share", "depth_encoder", "features")
    TEXT_PIPELINE_DIR: str = os.path.join(BASE_DIR, "member1_text_pipeline")
    
    # Output Directory
    OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
    CHECKPOINT_DIR: str = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")

    def __post_init__(self):
        if self.FEATURE_CHANNELS is None:
            self.FEATURE_CHANNELS = [80, 112, 160]
        if self.FEATURE_LAYERS is None:
            self.FEATURE_LAYERS = [7, 11, 13]
        for d in [self.CHECKPOINT_DIR, self.LOG_DIR]:
            os.makedirs(d, exist_ok=True)
            
    # Feature Dimensions
    FEATURE_DIM: int = 256
    HIDDEN_DIM: int = 512
    
    # GACM Settings (Phase 7)
    GACM_GATE_BIAS: float = 0.0
    
    # Training Hyperparameters (Phase 9)
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 100
    BATCH_SIZE: int = 16
    DEVICE: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"  # Auto-detect GPU
    SEED: int = 42
    DROPOUT: float = 0.1
    FEATURE_CHANNELS: List[int] = None  # Set in __post_init__
    ENCODER_PRETRAINED: bool = True
    FREEZE_UP_TO: int = 8
    FEATURE_LAYERS: List[int] = None  # Set in __post_init__

    def get_train_dir(self, category: str) -> str:
        return os.path.join(self.BASE_DIR, "mvtec_3d_anomaly_detection", category, "train", "good", "rgb")

cfg = FusionConfig()

