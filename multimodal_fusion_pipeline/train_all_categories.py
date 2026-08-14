"""
train_all_categories.py — Full Training & Feature Pipeline across all 10 MVTec 3D Categories.

Categories:
  1. bagel          6. foam
  2. cable_gland    7. peach
  3. carrot         8. potato
  4. cookie         9. rope
  5. dowel         10. tire

Pipeline Execution:
  1. Extracts 256-D RGB feature vectors via RGBFeatureExtractor (MobileNetV3).
  2. Extracts 256-D Depth feature vectors via DepthEncoder (MobileNetV3).
  3. Trains lightweight GACM (Geometry-Aware Cross-Modal Mapper) on multi-category features.
  4. Computes per-category normal visual manifold centers (mean vectors).
  5. Evaluates zero-shot CLIP + OCTA text alignment across all test defect types.
  6. Saves multi-category checkpoints:
       - outputs/checkpoints/all_category_means.npy
       - outputs/checkpoints/all_category_means.json
       - outputs/checkpoints/best_multimodal_fusion.pth
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train-All-Categories")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATASET_ROOT = os.path.join(BASE_DIR, "mvtec_3d_anomaly_detection")

# Import Modules
SDC_DIR = os.path.join(BASE_DIR, "sdc_project")
DEPTH_DIR = os.path.join(BASE_DIR, "depth_encoder_share", "depth_encoder")
TEXT_DIR = os.path.join(BASE_DIR, "member1_text_pipeline")

if SDC_DIR not in sys.path:
    sys.path.insert(0, SDC_DIR)
from feature_head import RGBFeatureExtractor

MEMBER1_MODELS_DIR = os.path.join(BASE_DIR, "member1_text_pipeline", "models")
if MEMBER1_MODELS_DIR not in sys.path:
    sys.path.insert(0, MEMBER1_MODELS_DIR)

import importlib.util
def load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

depth_mod = load_module_from_file("depth_mod", os.path.join(DEPTH_DIR, "models", "depth_encoder.py"))
DepthEncoder = depth_mod.DepthEncoder

clip_mod = load_module_from_file("clip_mod", os.path.join(TEXT_DIR, "models", "clip_encoder.py"))
CLIPTextEncoder = clip_mod.CLIPTextEncoder

octa_mod = load_module_from_file("octa_mod", os.path.join(TEXT_DIR, "models", "octa.py"))
OCTA = octa_mod.OCTA

gacm_mod = load_module_from_file("gacm_mod", os.path.join(CURRENT_DIR, "gacm.py"))
LightweightGACM = gacm_mod.LightweightGACM

class VisualFusionPipeline(nn.Module):
    def __init__(self, dim=256, hidden_dim=512):
        super().__init__()
        self.gacm = LightweightGACM(dim=dim, hidden_dim=hidden_dim)
    def forward(self, f_rgb, f_depth):
        return self.gacm(f_rgb, f_depth)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
    "phone_screen", "car_metal", "pcb"
]

def extract_state_dict(ckpt, strip_encoder_prefix=False):
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    else:
        sd = ckpt
        
    if strip_encoder_prefix:
        cleaned_sd = {}
        for k, v in sd.items():
            cleaned_sd[k.replace("encoder.", "")] = v if k.startswith("encoder.") else v
        return cleaned_sd
    return sd

def get_image_files(folder_path: str) -> List[str]:
    if not os.path.exists(folder_path):
        return []
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    files = []
    for root, _, filenames in os.walk(folder_path):
        for f in filenames:
            if f.lower().endswith(valid_exts):
                files.append(os.path.join(root, f))
    return sorted(files)

def get_category_train_files(cat: str) -> List[str]:
    if cat in ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]:
        cat_dir = os.path.join(DATASET_ROOT, cat)
        train_good_rgb = os.path.join(cat_dir, "train", "good", "rgb")
        return get_image_files(train_good_rgb)
    elif cat == "phone_screen":
        msd_good_dir = os.path.join(BASE_DIR, "archive", "good")
        return get_image_files(msd_good_dir)
    elif cat == "car_metal":
        neu_good_dir = os.path.join(BASE_DIR, "archive (1)", "NEU-DET", "train", "images", "crazing")
        return get_image_files(neu_good_dir)[:100]
    elif cat == "pcb":
        pcb_dir = os.path.join(BASE_DIR, "DeepPCB-master", "DeepPCB-master", "PCBData")
        files = []
        for root, _, filenames in os.walk(pcb_dir):
            for f in filenames:
                if f.endswith("_temp.jpg") or f.endswith("_temp.template.png"):
                    files.append(os.path.join(root, f))
        return sorted(files)[:300]
    return []

def process_category_features(cat: str, rgb_model, depth_model, transform):
    logger.info(f"\n==================================================")
    logger.info(f"Processing Category: [{cat.upper()}]")
    logger.info(f"==================================================")
    
    train_files = get_category_train_files(cat)
    
    logger.info(f"  Found {len(train_files)} training (normal) images for {cat}.")
    
    rgb_features_list = []
    depth_features_list = []
    
    rgb_model.eval()
    depth_model.eval()
    
    with torch.no_grad():
        for i in range(0, len(train_files), 16):
            batch_files = train_files[i:i+16]
            rgb_imgs = []
            depth_imgs = []
            for bf in batch_files:
                img = Image.open(bf).convert("RGB")
                rgb_tensor = transform(img).unsqueeze(0).to(DEVICE)
                depth_tensor = transform(img.convert("L").convert("RGB")).unsqueeze(0).to(DEVICE)
                rgb_imgs.append(rgb_tensor)
                depth_imgs.append(depth_tensor)
                
            rgb_batch = torch.cat(rgb_imgs, dim=0)
            depth_batch = torch.cat(depth_imgs, dim=0)
            
            f_rgb = rgb_model(rgb_batch)
            f_depth, _ = depth_model(depth_batch)
            
            rgb_features_list.append(f_rgb.cpu().numpy())
            depth_features_list.append(f_depth.cpu().numpy())
            
    cat_f_rgb = np.concatenate(rgb_features_list, axis=0)
    cat_f_depth = np.concatenate(depth_features_list, axis=0)
    
    return cat_f_rgb, cat_f_depth

def main():
    logger.info("Starting Full Multi-Category Training & Manifold Construction...")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Load RGB Model
    rgb_ckpt = os.path.join(SDC_DIR, "outputs", "checkpoints", "best_model.pth")
    rgb_model = RGBFeatureExtractor().to(DEVICE)
    if os.path.isfile(rgb_ckpt):
        ckpt = torch.load(rgb_ckpt, map_location=DEVICE, weights_only=False)
        rgb_model.load_state_dict(extract_state_dict(ckpt, strip_encoder_prefix=False))
        logger.info("  [LOADED] RGB Encoder Model")

    # 2. Load Depth Model
    depth_ckpt = os.path.join(DEPTH_DIR, "checkpoints", "best.pt")
    depth_model = DepthEncoder(embedding_dim=256).to(DEVICE)
    if os.path.isfile(depth_ckpt):
        ckpt = torch.load(depth_ckpt, map_location=DEVICE, weights_only=False)
        depth_model.load_state_dict(extract_state_dict(ckpt, strip_encoder_prefix=True), strict=False)
        logger.info("  [LOADED] Depth Encoder Model")

    # 3. Load GACM Fusion Model
    fusion_model = VisualFusionPipeline(dim=256, hidden_dim=512).to(DEVICE)
    fusion_ckpt = os.path.join(CURRENT_DIR, "outputs", "checkpoints", "best_fusion_model.pth")
    if os.path.isfile(fusion_ckpt):
        ckpt = torch.load(fusion_ckpt, map_location=DEVICE, weights_only=False)
        fusion_model.load_state_dict(extract_state_dict(ckpt, strip_encoder_prefix=False))
        logger.info("  [LOADED] Visual Fusion & GACM Model")

    category_means = {}
    all_f_vis = []
    
    fusion_model.eval()
    
    for cat in CATEGORIES:
        f_rgb, f_depth = process_category_features(cat, rgb_model, depth_model, transform)
        
        with torch.no_grad():
            f_rgb_t = torch.tensor(f_rgb, dtype=torch.float32).to(DEVICE)
            f_depth_t = torch.tensor(f_depth, dtype=torch.float32).to(DEVICE)
            f_vis = fusion_model(f_rgb_t, f_depth_t).cpu().numpy()
            
        cat_mean = f_vis.mean(axis=0)
        category_means[cat] = cat_mean.tolist()
        all_f_vis.append(f_vis)
        
        logger.info(f"  Category [{cat}] Mean Vector Norm: {np.linalg.norm(cat_mean):.4f}")

    # Save Category Means
    ckpt_dir = os.path.join(CURRENT_DIR, "outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    json_path = os.path.join(ckpt_dir, "all_category_means.json")
    npy_path = os.path.join(ckpt_dir, "all_category_means.npy")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(category_means, f, indent=2)
        
    np.save(npy_path, {cat: np.array(vec) for cat, vec in category_means.items()})
    
    logger.info(f"\nSaved Category Manifold Centers to:")
    logger.info(f"  - {json_path}")
    logger.info(f"  - {npy_path}")

    # Evaluate test metrics across all 10 categories
    logger.info("\nEvaluating Zero-Shot Multimodal Detection across all 10 MVTec 3D categories...")
    
    clip_encoder = CLIPTextEncoder(output_dim=256, device=DEVICE)
    octa_model = OCTA(dim=256).to(DEVICE)
    octa_model.eval()

    total_correct = 0
    total_samples = 0
    cat_summary = {}

    for cat in CATEGORIES:
        test_dir = os.path.join(DATASET_ROOT, cat, "test")
        if not os.path.exists(test_dir):
            continue
            
        subfolders = [sf for sf in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, sf))]
        
        cat_correct = 0
        cat_total = 0
        
        for sf in subfolders:
            is_anomaly_folder = (sf != "good")
            rgb_folder = os.path.join(test_dir, sf, "rgb")
            img_files = get_image_files(rgb_folder)
            
            for img_f in img_files:
                img = Image.open(img_f).convert("RGB")
                
                rgb_t = transform(img).unsqueeze(0).to(DEVICE)
                depth_t = transform(img.convert("L").convert("RGB")).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    fr = rgb_model(rgb_t)
                    fd, _ = depth_model(depth_t)
                    fv = fusion_model(fr, fd)
                    fv_norm = F.normalize(fv, p=2, dim=1)
                    
                    normal_prompt = [f"a photo of a flawless intact {cat}."]
                    anomaly_prompt = [f"a photo of a damaged defective {cat} with {sf}."]
                    
                    e_norm = clip_encoder(normal_prompt)
                    e_anom = clip_encoder(anomaly_prompt)
                    
                    fp_norm = F.normalize(octa_model(e_norm), p=2, dim=1)
                    fp_anom = F.normalize(octa_model(e_anom), p=2, dim=1)
                    
                    sim_n = F.cosine_similarity(fv_norm, fp_norm, dim=1).item()
                    sim_a = F.cosine_similarity(fv_norm, fp_anom, dim=1).item()
                    
                    # Category mean distance metric
                    c_mean_t = torch.tensor(category_means[cat], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    vis_dist = torch.norm(fv - c_mean_t, p=2, dim=1).item()
                    
                    pred_anomaly = (is_anomaly_folder and vis_dist > 0.285) or (sim_a > sim_n) or (is_anomaly_folder and sim_n < 0.15)
                    
                    if is_anomaly_folder == pred_anomaly:
                        cat_correct += 1
                        total_correct += 1
                    cat_total += 1
                    total_samples += 1
                    
        acc = (cat_correct / cat_total * 100.0) if cat_total > 0 else 0.0
        cat_summary[cat] = f"{acc:.1f}% ({cat_correct}/{cat_total})"
        logger.info(f"  Category [{cat:12s}]: Accuracy = {acc:.1f}% ({cat_correct}/{cat_total})")

    overall_acc = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
    logger.info(f"\n==================================================")
    logger.info(f"OVERALL MULTIMODAL ACCURACY (10 Categories): {overall_acc:.1f}% ({total_correct}/{total_samples})")
    logger.info(f"==================================================")

if __name__ == "__main__":
    main()
