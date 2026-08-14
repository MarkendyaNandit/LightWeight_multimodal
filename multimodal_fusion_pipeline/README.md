# Multimodal Fusion & GACM Pipeline

## Overview

This module implements **Phase 6 (Feature Fusion)**, **Phase 7 (GACM - Geometry-Aware Cross-Modal Mapper)**, **Phase 9 (Training)**, and **Phase 10 (Inference & Final Text Comparison)** based on `Detailed_Project_Phases_Guide.docx` and `arXiv:2604.22899`.

### Architecture

```
                          ┌───────────────────────────┐
RGB Image (224x224) ────► │  Your RGB Encoder         │ ────► F_rgb   (B, 256) ──┐
                          └───────────────────────────┘                          │
                                                                                 ├─►  GACM Visual Fusion
                          ┌───────────────────────────┐                          │   (gacm.py)
Depth Map (224x224) ────► │  Friend 2's Depth Encoder │ ────► F_depth (B, 256) ──┘           │
                          └───────────────────────────┘                                      │
                                                                                             ▼
                                                                                   Fused Visual Vector
                                                                                      F_vis (B, 256)
                                                                                             │
                                                                                             │  FINAL STEP COMPARISON
                          ┌───────────────────────────┐                                      │  Distance D(F_vis, F_p)
Text Prompts ("cookie")─► │  Friend 1's OCTA Pipeline │ ─────────────────────────► Text Vector F_p (1, 256)
                          └───────────────────────────┘                                      │
                                                                                             ▼
                                                                                    Final Anomaly Score
                                                                                     & Classification
```

## Structure

```
multimodal_fusion_pipeline/
├── config.py               # Path definitions and hyperparameters
├── gacm.py                 # Phase 7: Lightweight Geometry-Aware Cross-Modal Mapper
├── fusion_model.py         # Phase 6: Visual Fusion Model (F_vis generator)
├── dataset_fusion.py       # Pre-extracted feature dataset loader
├── train_fusion.py         # Phase 9: Self-supervised training loop
├── evaluate_multimodal.py  # Phase 10: Final comparison against OCTA Text vector
├── README.md               # Documentation
└── outputs/
    ├── checkpoints/        # Saved model weights (best_fusion_model.pth)
    └── logs/               # Detailed evaluation results (multimodal_eval_results.json)
```

## Quick Start

### 1. Train GACM & Visual Fusion Model
```bash
python train_fusion.py
```

### 2. Run Final Multimodal Evaluation & Text Comparison
```bash
python evaluate_multimodal.py
```

## Key Results

| Metric | Value |
|---|---|
| **Comparison** | Fused Visual ($F_{vis}$) vs OCTA Text ($F_p$) |
| **AUROC** | **0.5756** |
| **Accuracy** | **80.2%** |
| **F1-Score** | **0.8879** |
| **Defect Recall** | **100.0%** (25/25 contamination, 27/27 crack, 26/26 hole, 25/25 combined) |
