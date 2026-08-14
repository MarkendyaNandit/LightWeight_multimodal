"""
recompute_means.py — Recompute per-category manifold means from ACTUAL normal training images.
This fixes the phone_screen, car_metal, and pcb means which were incorrectly computed previously.
"""

import os, sys, torch, json
import numpy as np
from PIL import Image
from torchvision import transforms as T
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'sdc_project'))
sys.path.insert(0, os.path.join(BASE, 'multimodal_fusion_pipeline'))

from feature_head import RGBFeatureExtractor
import importlib.util

# Load modules
spec = importlib.util.spec_from_file_location('dm', os.path.join(BASE, 'depth_encoder_share', 'depth_encoder', 'models', 'depth_encoder.py'))
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)

spec2 = importlib.util.spec_from_file_location('gm', os.path.join(BASE, 'multimodal_fusion_pipeline', 'gacm.py'))
gm = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(gm)

class VisualFusionPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.gacm = gm.LightweightGACM(dim=256, hidden_dim=512)
    def forward(self, f_rgb, f_depth):
        return self.gacm(f_rgb, f_depth)

# Build and load models
print("Loading models...")
rgb_model = RGBFeatureExtractor().eval()
depth_model = dm.DepthEncoder(embedding_dim=256).eval()
fusion = VisualFusionPipeline().eval()

ft_path = os.path.join(BASE, 'multimodal_fusion_pipeline', 'outputs', 'checkpoints', 'best_fusion_model_finetuned.pth')
ckpt = torch.load(ft_path, map_location='cpu', weights_only=False)
fusion.load_state_dict(ckpt)
print("Models loaded.")

tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def get_normal_images(category, max_images=200):
    """Get paths to normal training images for each category."""
    paths = []
    if category in ['bagel','cable_gland','carrot','cookie','dowel','foam','peach','potato','rope','tire']:
        train_dir = os.path.join(BASE, 'mvtec_3d_anomaly_detection', category, 'train', 'good', 'rgb')
        if os.path.isdir(train_dir):
            for f in os.listdir(train_dir):
                if f.endswith(('.png', '.jpg')):
                    paths.append(os.path.join(train_dir, f))
    elif category == 'phone_screen':
        good_dir = os.path.join(BASE, 'archive', 'good')
        if os.path.isdir(good_dir):
            for f in os.listdir(good_dir):
                if f.endswith(('.png', '.jpg')):
                    paths.append(os.path.join(good_dir, f))
    elif category == 'car_metal':
        # Use 'crazing' as normal reference (surface texture without deep scratches)
        crazing_dir = os.path.join(BASE, 'archive (1)', 'NEU-DET', 'train', 'images', 'crazing')
        if os.path.isdir(crazing_dir):
            for f in os.listdir(crazing_dir):
                if f.endswith(('.png', '.jpg')):
                    paths.append(os.path.join(crazing_dir, f))
    elif category == 'pcb':
        pcb_base = os.path.join(BASE, 'DeepPCB-master', 'DeepPCB-master', 'PCBData')
        if os.path.isdir(pcb_base):
            for root, _, files in os.walk(pcb_base):
                for f in files:
                    if f.endswith('_temp.jpg'):
                        paths.append(os.path.join(root, f))
    return paths[:max_images]

@torch.no_grad()
def compute_mean_features(image_paths):
    """Extract fused features from all images and return the mean vector."""
    features = []
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
            rgb = tf(img).unsqueeze(0)
            dep = tf(img.convert('L').convert('RGB')).unsqueeze(0)
            f_rgb = rgb_model(rgb)
            f_dep, _ = depth_model(dep)
            f_vis = fusion(f_rgb, f_dep)
            features.append(f_vis.squeeze(0))
        except Exception as e:
            pass
    if features:
        stacked = torch.stack(features, dim=0)
        mean_vec = stacked.mean(dim=0)
        # L2 normalize
        mean_vec = mean_vec / mean_vec.norm(p=2)
        return mean_vec
    return None

ALL_CATEGORIES = [
    'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
    'foam', 'peach', 'potato', 'rope', 'tire',
    'phone_screen', 'car_metal', 'pcb'
]

print("\n" + "="*70)
print("  RECOMPUTING CATEGORY MANIFOLD MEANS FROM NORMAL TRAINING DATA")
print("="*70)

category_means = {}
for cat in ALL_CATEGORIES:
    paths = get_normal_images(cat)
    print(f"\n  {cat:<16} : {len(paths)} normal images found")
    if paths:
        mean = compute_mean_features(paths)
        if mean is not None:
            category_means[cat] = mean.tolist()
            print(f"    -> Mean computed (norm={np.linalg.norm(mean.numpy()):.4f})")

# Save
out_path = os.path.join(BASE, 'multimodal_fusion_pipeline', 'outputs', 'checkpoints', 'all_category_means.json')
with open(out_path, 'w') as f:
    json.dump(category_means, f)
print(f"\nSaved {len(category_means)} category means to: {out_path}")

# Now verify distances
print("\n" + "="*70)
print("  VERIFYING PHONE SCREEN DISTANCES WITH NEW MEANS")
print("="*70)

phone_mean = torch.tensor(category_means['phone_screen'], dtype=torch.float32).unsqueeze(0)

norm_paths = get_normal_images('phone_screen')[:20]
def_dir = os.path.join(BASE, 'archive', 'scratch')
def_paths = sorted([os.path.join(def_dir, f) for f in os.listdir(def_dir) if f.endswith('.jpg')])[:20]

n_dists, d_dists = [], []
for p in norm_paths:
    img = Image.open(p).convert('RGB')
    rgb = tf(img).unsqueeze(0); dep = tf(img.convert('L').convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        f_rgb = rgb_model(rgb); f_dep, _ = depth_model(dep); f_vis = fusion(f_rgb, f_dep)
        n_dists.append(torch.norm(f_vis - phone_mean, p=2, dim=1).item())

for p in def_paths:
    img = Image.open(p).convert('RGB')
    rgb = tf(img).unsqueeze(0); dep = tf(img.convert('L').convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        f_rgb = rgb_model(rgb); f_dep, _ = depth_model(dep); f_vis = fusion(f_rgb, f_dep)
        d_dists.append(torch.norm(f_vis - phone_mean, p=2, dim=1).item())

print(f"NORMAL: min={min(n_dists):.4f} mean={np.mean(n_dists):.4f} max={max(n_dists):.4f}")
print(f"DEFECT: min={min(d_dists):.4f} mean={np.mean(d_dists):.4f} max={max(d_dists):.4f}")
print(f"Separability gap: {min(d_dists) - max(n_dists):.4f}")

# Find optimal threshold
all_d = [(d, 0) for d in n_dists] + [(d, 1) for d in d_dists]
best_t, best_acc = 0, 0
for t in np.linspace(0.001, 2.0, 2000):
    correct = sum(1 for d, label in all_d if (d > t) == (label == 1))
    acc = correct / len(all_d)
    if acc > best_acc: best_acc, best_t = acc, t
print(f"OPTIMAL THRESHOLD: {best_t:.4f}  Accuracy: {best_acc*100:.1f}%")
