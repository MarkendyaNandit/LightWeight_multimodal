"""
build_patchcore_banks.py — Memory Bank Generator for PatchCore

Scans normal training images across all 13 categories, extracts 196 patch vectors
per image, applies K-Center Greedy coreset reduction (10% sampling ratio),
and saves the lightweight coreset dictionary to `patchcore_coreset_banks.pt`.
"""

import os, sys, json, logging, time
from typing import Dict, List

import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildPatchCore")

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(CURRENT_DIR)
DATASET_ROOT = os.path.join(BASE_DIR, "mvtec_3d_anomaly_detection")
SDC_DIR      = os.path.join(BASE_DIR, "sdc_project")
DEPTH_DIR    = os.path.join(BASE_DIR, "depth_encoder_share", "depth_encoder")
CKPT_DIR     = os.path.join(CURRENT_DIR, "outputs", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

for p in [SDC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

from feature_head import RGBFeatureExtractor
depth_mod       = load_module("depth_mod",  os.path.join(DEPTH_DIR, "models", "depth_encoder.py"))
DepthEncoder    = depth_mod.DepthEncoder
gacm_mod        = load_module("gacm_mod",   os.path.join(CURRENT_DIR, "gacm.py"))
LightweightGACM = gacm_mod.LightweightGACM

import patchcore_engine
MultimodalPatchExtractor = patchcore_engine.MultimodalPatchExtractor
KCenterGreedyCoreset     = patchcore_engine.KCenterGreedyCoreset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE}")

CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
    "phone_screen", "car_metal", "pcb"
]

def _scan(folder):
    valid = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    found = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(valid):
                found.append(os.path.join(root, f))
    return sorted(found)

def get_normal_files(cat):
    if cat in ["bagel","cable_gland","carrot","cookie","dowel","foam","peach","potato","rope","tire"]:
        return _scan(os.path.join(DATASET_ROOT, cat, "train", "good", "rgb"))
    elif cat == "phone_screen":
        files = _scan(os.path.join(BASE_DIR, "archive", "good"))
        for sub in ["scratch", "oil", "stain"]:
            files += _scan(os.path.join(BASE_DIR, "archive", sub))
        return files[:200]
    elif cat == "car_metal":
        files = []
        for sub in ["crazing", "patches", "rolled-in_scale"]:
            files += _scan(os.path.join(BASE_DIR, "archive (1)", "NEU-DET", "train", "images", sub))
        return files[:200]
    elif cat == "pcb":
        files = []
        for root, _, fns in os.walk(os.path.join(BASE_DIR, "DeepPCB-master", "DeepPCB-master", "PCBData")):
            for f in fns:
                if f.endswith("_temp.jpg"):
                    files.append(os.path.join(root, f))
        return sorted(files)[:200]
    return []

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(**NORM)])

def safe_open(p):
    try:    return Image.open(p).convert("RGB")
    except: return Image.new("RGB", (224, 224))

def get_depth_image(rgb_path, cat):
    if cat in ["bagel","cable_gland","carrot","cookie","dowel","foam","peach","potato","rope","tire"]:
        xyz_path = rgb_path.replace(os.sep+"rgb"+os.sep, os.sep+"xyz"+os.sep).replace(".png",".tiff").replace(".jpg",".tiff")
        if os.path.isfile(xyz_path):
            try:
                img = Image.open(xyz_path)
                arr = np.array(img).astype(np.float32)
                ch  = arr[:,:,2] if arr.ndim == 3 else arr
                ch  = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                return Image.fromarray((ch * 255).astype(np.uint8)).convert("RGB")
            except: pass
    return safe_open(rgb_path).convert("L").convert("RGB")

def extract_sd(ckpt, strip=False):
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    else: sd = ckpt
    if strip: sd = {k.replace("encoder.",""): v for k,v in sd.items()}
    return sd

def load_models():
    rgb_model = RGBFeatureExtractor().to(DEVICE)
    ck = os.path.join(SDC_DIR, "outputs", "checkpoints", "best_model.pth")
    if os.path.isfile(ck):
        rgb_model.load_state_dict(extract_sd(torch.load(ck, map_location=DEVICE, weights_only=False)))

    depth_model = DepthEncoder(embedding_dim=256).to(DEVICE)
    ck = os.path.join(DEPTH_DIR, "checkpoints", "best.pt")
    if os.path.isfile(ck):
        depth_model.load_state_dict(extract_sd(torch.load(ck, map_location=DEVICE, weights_only=False), strip=True), strict=False)

    import torch.nn as nn
    class VisualFusionPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.gacm = LightweightGACM(dim=256, hidden_dim=512)
        def forward(self, fr, fd): return self.gacm(fr, fd)

    fusion_model = VisualFusionPipeline().to(DEVICE)
    ck = os.path.join(CKPT_DIR, "best_fusion_model_finetuned.pth")
    if not os.path.isfile(ck):
        ck = os.path.join(CKPT_DIR, "best_fusion_model.pth")
    if os.path.isfile(ck):
        fusion_model.load_state_dict(extract_sd(torch.load(ck, map_location=DEVICE, weights_only=False)))

    for m in [rgb_model, depth_model, fusion_model]:
        for p in m.parameters(): p.requires_grad = False
        m.eval()

    return rgb_model, depth_model, fusion_model

def main():
    logger.info("="*60)
    logger.info("  BUILDING MULTIMODAL PATCHCORE CORESET MEMORY BANKS (13 Categories)")
    logger.info("="*60)
    t0 = time.time()

    rgb_model, depth_model, fusion_model = load_models()
    extractor = MultimodalPatchExtractor(rgb_model, depth_model, fusion_model, target_size=(14,14))
    coreset_sampler = KCenterGreedyCoreset(sampling_ratio=0.10)  # 10% coreset size

    coreset_banks = {}

    for cat in CATEGORIES:
        files = get_normal_files(cat)
        if not files:
            logger.warning(f"  [{cat}] No normal files found!")
            continue

        logger.info(f"\n  [{cat.upper()}] Extracting patches from {len(files)} normal images...")
        all_patches = []

        # Process images in mini-batches
        for i in range(0, len(files), 32):
            batch_files = files[i:i+32]
            rgb_imgs, dep_imgs = [], []
            for p in batch_files:
                rgb_img = safe_open(p)
                dep_img = get_depth_image(p, cat)
                rgb_imgs.append(transform(rgb_img).unsqueeze(0))
                dep_imgs.append(transform(dep_img).unsqueeze(0))

            rgb_b = torch.cat(rgb_imgs, 0).to(DEVICE)
            dep_b = torch.cat(dep_imgs, 0).to(DEVICE)

            # (B, 196, 256)
            patch_vecs = extractor.extract_patch_features(rgb_b, dep_b)
            # Flatten to (B*196, 256)
            B, P, D = patch_vecs.shape
            all_patches.append(patch_vecs.reshape(B * P, D).cpu())

        # Concatenate all normal patches for this category: (N_total_patches, 256)
        cat_memory_bank = torch.cat(all_patches, 0)
        logger.info(f"    Raw memory bank: {cat_memory_bank.shape[0]} patch vectors")

        # Subsample to coreset memory bank using K-Center Greedy
        coreset_bank = coreset_sampler.sample(cat_memory_bank)
        coreset_banks[cat] = coreset_bank
        logger.info(f"    Saved coreset bank: {coreset_bank.shape[0]} vectors  (dtype={coreset_bank.dtype})")

    # Save all coreset memory banks into a single checkpoint
    out_path = os.path.join(CKPT_DIR, "patchcore_coreset_banks.pt")
    torch.save(coreset_banks, out_path)
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)

    logger.info("\n" + "="*60)
    logger.info(f"  PATCHCORE MEMORY BANKS BUILT SUCCESSFULLY!")
    logger.info(f"  Checkpoint saved to: {out_path}")
    logger.info(f"  Total memory bank size: {file_size_mb:.1f} MB (fits easily on HF Free Space)")
    logger.info(f"  Total time elapsed: {(time.time()-t0)/60:.1f} min")
    logger.info("="*60)

if __name__ == "__main__":
    main()
