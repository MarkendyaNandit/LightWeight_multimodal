"""
train_cached.py — Fast Cache-First GACM Fine-Tuning (All 13 Categories, CPU-Optimised)

Strategy:
  PHASE 1 (one-time): Extract ALL features from frozen RGB + Depth encoders into RAM.
                      Image I/O happens only ONCE — not every epoch.
  PHASE 2 (fast):     Train GACM purely on pre-cached feature tensors.
                      Each epoch is ~50x faster than loading images every step.

Trains GACM fusion head with:
  - NT-Xent contrastive loss  (normal vs augmented-defect pairs in feature space)
  - Category Center Loss      (pull normal embeddings toward their class centroid)
  - AdamW + Cosine Annealing  LR scheduler
  - Early stopping            (patience = 5 epochs)

After training:
  - Recomputes all 13 category manifold means with the best GACM weights
  - Saves best_fusion_model_finetuned.pth + all_category_means.json + training_history.json
"""

import os, sys, json, logging, time, random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms as T
from PIL import Image

# ─── Config ───────────────────────────────────────────────
NUM_EPOCHS   = 30
BATCH_SIZE   = 128        # large batches fine since all tensors are in RAM
LR           = 1e-4
PATIENCE     = 5
TEMP         = 0.07
WEIGHT_DECAY = 1e-4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train-Cached")

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(CURRENT_DIR)
DATASET_ROOT = os.path.join(BASE_DIR, "mvtec_3d_anomaly_detection")
SDC_DIR      = os.path.join(BASE_DIR, "sdc_project")
DEPTH_DIR    = os.path.join(BASE_DIR, "depth_encoder_share", "depth_encoder")
CKPT_DIR     = os.path.join(CURRENT_DIR, "outputs", "checkpoints")
CACHE_DIR    = os.path.join(CURRENT_DIR, "outputs", "feature_cache")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE}")

# ─── Categories ───────────────────────────────────────────
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
        return files[:300]
    elif cat == "car_metal":
        files = []
        for sub in ["crazing", "patches", "rolled-in_scale"]:
            files += _scan(os.path.join(BASE_DIR, "archive (1)", "NEU-DET", "train", "images", sub))
        return files[:300]
    elif cat == "pcb":
        files = []
        for root, _, fns in os.walk(os.path.join(BASE_DIR, "DeepPCB-master", "DeepPCB-master", "PCBData")):
            for f in fns:
                if f.endswith("_temp.jpg"):
                    files.append(os.path.join(root, f))
        return sorted(files)[:400]
    return []

def get_defect_files(cat):
    if cat in ["bagel","cable_gland","carrot","cookie","dowel","foam","peach","potato","rope","tire"]:
        test_dir = os.path.join(DATASET_ROOT, cat, "test")
        files = []
        if os.path.isdir(test_dir):
            for sub in os.listdir(test_dir):
                if sub != "good" and os.path.isdir(os.path.join(test_dir, sub)):
                    files += _scan(os.path.join(test_dir, sub, "rgb"))
        return files
    elif cat == "phone_screen":
        files = []
        for sub in ["scratch", "oil", "stain"]:
            files += _scan(os.path.join(BASE_DIR, "archive", sub))
        return files
    elif cat == "car_metal":
        files = []
        for sub in ["inclusion", "pitted_surface", "scratches"]:
            files += _scan(os.path.join(BASE_DIR, "archive (1)", "NEU-DET", "train", "images", sub))
        return files[:300]
    elif cat == "pcb":
        files = []
        for root, _, fns in os.walk(os.path.join(BASE_DIR, "DeepPCB-master", "DeepPCB-master", "PCBData")):
            for f in fns:
                if f.endswith("_test.jpg"):
                    files.append(os.path.join(root, f))
        return sorted(files)[:400]
    return []

# ─── Transform ────────────────────────────────────────────
NORM = dict(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
base_transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(**NORM)])

def safe_open(p):
    try:    return Image.open(p).convert("RGB")
    except: return Image.new("RGB",(224,224))

# ─── Model helpers ────────────────────────────────────────
class VisualFusionPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.gacm = LightweightGACM(dim=256, hidden_dim=512)
    def forward(self, fr, fd): return self.gacm(fr, fd)

def extract_sd(ckpt, strip=False):
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    else:
        sd = ckpt
    if strip:
        sd = {k.replace("encoder.",""): v for k,v in sd.items()}
    return sd

def load_encoders():
    rgb_model = RGBFeatureExtractor().to(DEVICE)
    ck = os.path.join(SDC_DIR, "outputs","checkpoints","best_model.pth")
    if os.path.isfile(ck):
        rgb_model.load_state_dict(extract_sd(torch.load(ck,map_location=DEVICE,weights_only=False)))
        logger.info("  [LOADED] RGB Encoder")

    depth_model = DepthEncoder(embedding_dim=256).to(DEVICE)
    ck = os.path.join(DEPTH_DIR,"checkpoints","best.pt")
    if os.path.isfile(ck):
        depth_model.load_state_dict(extract_sd(torch.load(ck,map_location=DEVICE,weights_only=False),strip=True), strict=False)
        logger.info("  [LOADED] Depth Encoder")

    for p in rgb_model.parameters():   p.requires_grad = False
    for p in depth_model.parameters(): p.requires_grad = False
    rgb_model.eval(); depth_model.eval()
    return rgb_model, depth_model

def load_fusion():
    fusion = VisualFusionPipeline().to(DEVICE)
    ck = os.path.join(CKPT_DIR, "best_fusion_model_finetuned.pth")
    if not os.path.isfile(ck):
        ck = os.path.join(CKPT_DIR, "best_fusion_model.pth")
    if os.path.isfile(ck):
        fusion.load_state_dict(extract_sd(torch.load(ck,map_location=DEVICE,weights_only=False)))
        logger.info(f"  [LOADED] GACM from {os.path.basename(ck)}")
    else:
        logger.info("  [NEW] GACM — random init")
    return fusion

# ─── PHASE 1: Pre-extract all features ────────────────────
@torch.no_grad()
def extract_all_features(cat_normal_map, cat_defect_map, rgb_model, depth_model):
    """
    Extract RGB + Depth features for ALL images once. Returns:
      norm_feats_rgb:   (N_normal, 256) tensor
      norm_feats_dep:   (N_normal, 256) tensor
      norm_cat_ids:     (N_normal,)     tensor
      def_feats_rgb:    (N_defect, 256) tensor
      def_feats_dep:    (N_defect, 256) tensor
      def_cat_ids:      (N_defect,)     tensor
    """
    cat_to_id = {c:i for i,c in enumerate(CATEGORIES)}

    def extract_batch(file_list, cat_id):
        all_rgb, all_dep, all_ids = [], [], []
        for i in range(0, len(file_list), 64):
            batch_paths = file_list[i:i+64]
            ri, di = [], []
            for p in batch_paths:
                img = safe_open(p)
                ri.append(base_transform(img).unsqueeze(0))
                di.append(base_transform(img.convert("L").convert("RGB")).unsqueeze(0))
            rb = torch.cat(ri,0).to(DEVICE)
            db = torch.cat(di,0).to(DEVICE)
            fr  = rgb_model(rb)
            fd, _ = depth_model(db)
            all_rgb.append(fr.cpu())
            all_dep.append(fd.cpu())
            all_ids.append(torch.full((len(batch_paths),), cat_id, dtype=torch.long))
        return all_rgb, all_dep, all_ids

    norm_rgb_all, norm_dep_all, norm_ids_all = [], [], []
    def_rgb_all,  def_dep_all,  def_ids_all  = [], [], []

    for cat in CATEGORIES:
        cid = cat_to_id[cat]

        normal_files = cat_normal_map[cat]
        if normal_files:
            logger.info(f"  Extracting normals  [{cat:15s}] {len(normal_files)} imgs")
            rr, rd, ri = extract_batch(normal_files, cid)
            norm_rgb_all.extend(rr); norm_dep_all.extend(rd); norm_ids_all.extend(ri)

        defect_files = cat_defect_map[cat]
        if defect_files:
            logger.info(f"  Extracting defects  [{cat:15s}] {len(defect_files)} imgs")
            rr, rd, ri = extract_batch(defect_files, cid)
            def_rgb_all.extend(rr);  def_dep_all.extend(rd);  def_ids_all.extend(ri)

    norm_rgb = torch.cat(norm_rgb_all, 0)
    norm_dep = torch.cat(norm_dep_all, 0)
    norm_ids = torch.cat(norm_ids_all, 0)
    def_rgb  = torch.cat(def_rgb_all,  0)
    def_dep  = torch.cat(def_dep_all,  0)
    def_ids  = torch.cat(def_ids_all,  0)

    logger.info(f"  Normal features: {norm_rgb.shape}  Defect features: {def_rgb.shape}")
    return norm_rgb, norm_dep, norm_ids, def_rgb, def_dep, def_ids

# ─── Losses ───────────────────────────────────────────────
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__(); self.temp = temperature
    def forward(self, z_a, z_p, z_n):
        z_a = F.normalize(z_a,p=2,dim=1)
        z_p = F.normalize(z_p,p=2,dim=1)
        z_n = F.normalize(z_n,p=2,dim=1)
        logits = torch.stack([(z_a*z_p).sum(1)/self.temp,
                               (z_a*z_n).sum(1)/self.temp], dim=1)
        return F.cross_entropy(logits, torch.zeros(z_a.size(0),dtype=torch.long,device=z_a.device))

class CatCenterLoss(nn.Module):
    def __init__(self, num_cats, dim=256):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_cats, dim))
    def forward(self, z, cat_ids):
        return (1-(F.normalize(z,p=2,dim=1)*F.normalize(self.centers[cat_ids],p=2,dim=1)).sum(1)).mean()

# ─── Compute manifold means after training ────────────────
@torch.no_grad()
def recompute_means(cat_normal_map, rgb_model, depth_model, fusion_model):
    fusion_model.eval(); rgb_model.eval(); depth_model.eval()
    out = {}
    for cat, files in cat_normal_map.items():
        if not files: continue
        vecs = []
        for i in range(0, len(files), 64):
            batch = files[i:i+64]
            ri, di = [], []
            for p in batch:
                img = safe_open(p)
                ri.append(base_transform(img).unsqueeze(0))
                di.append(base_transform(img.convert("L").convert("RGB")).unsqueeze(0))
            fr  = rgb_model(torch.cat(ri,0).to(DEVICE))
            fd, _ = depth_model(torch.cat(di,0).to(DEVICE))
            vecs.append(fusion_model(fr, fd).cpu())
        all_v = torch.cat(vecs,0).numpy()
        mean  = all_v.mean(0)
        out[cat] = mean.tolist()
        logger.info(f"  [{cat:15s}] norm={np.linalg.norm(mean):.4f}  n={len(files)}")
    return out

# ─── Main ─────────────────────────────────────────────────
def main():
    logger.info("="*60)
    logger.info("  CACHE-FIRST GACM FINE-TUNING  (13 Categories, CPU-Optimised)")
    logger.info(f"  Epochs={NUM_EPOCHS}  BatchSize={BATCH_SIZE}  LR={LR}  Patience={PATIENCE}")
    logger.info("="*60)

    # Scan datasets
    logger.info("\n[1/5] Scanning datasets...")
    cat_normal_map, cat_defect_map = {}, {}
    for cat in CATEGORIES:
        n = get_normal_files(cat)
        d = get_defect_files(cat)
        cat_normal_map[cat] = n
        cat_defect_map[cat] = d
        logger.info(f"  {cat:15s}  normal={len(n):4d}  defect={len(d):4d}")

    # Load encoders
    logger.info("\n[2/5] Loading frozen encoders...")
    rgb_model, depth_model = load_encoders()

    # Check if cache exists
    cache_norm = os.path.join(CACHE_DIR, "norm_features.pt")
    cache_def  = os.path.join(CACHE_DIR, "def_features.pt")

    if os.path.isfile(cache_norm) and os.path.isfile(cache_def):
        logger.info("\n[3/5] Loading cached features from disk...")
        norm_data = torch.load(cache_norm, weights_only=False)
        def_data  = torch.load(cache_def,  weights_only=False)
        norm_rgb, norm_dep, norm_ids = norm_data["rgb"], norm_data["dep"], norm_data["ids"]
        def_rgb,  def_dep,  def_ids  = def_data["rgb"],  def_data["dep"],  def_data["ids"]
        logger.info(f"  Normal: {norm_rgb.shape}  Defect: {def_rgb.shape}")
    else:
        logger.info("\n[3/5] Extracting features (one-time, ~5-10 min)...")
        t0 = time.time()
        norm_rgb, norm_dep, norm_ids, def_rgb, def_dep, def_ids = \
            extract_all_features(cat_normal_map, cat_defect_map, rgb_model, depth_model)
        logger.info(f"  Extraction done in {(time.time()-t0)/60:.1f} min")
        torch.save({"rgb": norm_rgb, "dep": norm_dep, "ids": norm_ids}, cache_norm)
        torch.save({"rgb": def_rgb,  "dep": def_dep,  "ids": def_ids},  cache_def)
        logger.info(f"  Features cached to {CACHE_DIR}")

    # Build triplet dataset: (norm_rgb, norm_dep, def_rgb, def_dep, cat_id)
    logger.info("\n[4/5] Building in-memory triplet dataset...")
    N = norm_rgb.shape[0]
    D = def_rgb.shape[0]
    cat_to_def = {}
    for i in range(D):
        cid = def_ids[i].item()
        cat_to_def.setdefault(cid, []).append(i)

    # For each normal, pick a matching-category defect
    matched_def_idx = []
    for i in range(N):
        cid = norm_ids[i].item()
        pool = cat_to_def.get(cid, list(range(D)))
        matched_def_idx.append(random.choice(pool))

    matched_def_idx = torch.tensor(matched_def_idx, dtype=torch.long)
    dataset = TensorDataset(
        norm_rgb, norm_dep,
        def_rgb[matched_def_idx], def_dep[matched_def_idx],
        norm_ids
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    logger.info(f"  {N} normal samples  |  {len(loader)} steps/epoch")

    # Load fusion model
    fusion_model = load_fusion()

    # Optimizers & losses
    optimizer    = torch.optim.AdamW(fusion_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    ntxent_loss  = NTXentLoss(temperature=TEMP)
    center_loss  = CatCenterLoss(len(CATEGORIES)).to(DEVICE)
    center_optim = torch.optim.Adam(center_loss.parameters(), lr=1e-3)

    best_loss = float("inf"); patience_ct = 0; history = []

    logger.info(f"\n[5/5] Training GACM for up to {NUM_EPOCHS} epochs...")
    logger.info("="*60)

    for epoch in range(1, NUM_EPOCHS+1):
        fusion_model.train()
        e_loss = e_ntx = e_ctr = 0.0
        t0 = time.time()

        for fr_a, fd_a, fr_n, fd_n, cat_ids in loader:
            fr_a = fr_a.to(DEVICE); fd_a = fd_a.to(DEVICE)
            fr_p = fr_a + 0.02 * torch.randn_like(fr_a)   # tiny jitter = augmented positive
            fd_p = fd_a + 0.02 * torch.randn_like(fd_a)
            fr_n = fr_n.to(DEVICE); fd_n = fd_n.to(DEVICE)
            cat_ids = cat_ids.to(DEVICE)

            z_a = fusion_model(fr_a, fd_a)
            z_p = fusion_model(fr_p, fd_p)
            z_n = fusion_model(fr_n, fd_n)

            l_ntx = ntxent_loss(z_a, z_p, z_n)
            l_ctr = center_loss(z_a, cat_ids)
            loss  = l_ntx + 0.3 * l_ctr

            optimizer.zero_grad(); center_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step(); center_optim.step()

            e_loss += loss.item(); e_ntx += l_ntx.item(); e_ctr += l_ctr.item()

        scheduler.step()
        n  = max(len(loader), 1)
        al = e_loss/n; an = e_ntx/n; ac = e_ctr/n
        lr = scheduler.get_last_lr()[0]
        history.append({"epoch":epoch,"loss":al,"ntxent":an,"center":ac})

        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
            f"Loss={al:.4f}  NT-Xent={an:.4f}  Center={ac:.4f}  "
            f"LR={lr:.2e}  Time={time.time()-t0:.1f}s"
        )

        if al < best_loss:
            best_loss = al; patience_ct = 0
            sp = os.path.join(CKPT_DIR, "best_fusion_model_finetuned.pth")
            torch.save(fusion_model.state_dict(), sp)
            logger.info(f"  BEST saved -> {os.path.basename(sp)}")
        else:
            patience_ct += 1
            logger.info(f"  No improvement ({patience_ct}/{PATIENCE})")
            if patience_ct >= PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch}.")
                break

    logger.info(f"\nTraining complete. Best loss={best_loss:.4f}")

    # Load best weights
    bp = os.path.join(CKPT_DIR, "best_fusion_model_finetuned.pth")
    if os.path.isfile(bp):
        fusion_model.load_state_dict(torch.load(bp, map_location=DEVICE, weights_only=False))
        logger.info("  Best GACM weights loaded.")

    # Recompute manifold means with best weights
    logger.info("\nRecomputing 13-category manifold centers with best GACM weights...")
    means = recompute_means(cat_normal_map, rgb_model, depth_model, fusion_model)
    jp = os.path.join(CKPT_DIR, "all_category_means.json")
    np_ = os.path.join(CKPT_DIR, "all_category_means.npy")
    with open(jp,"w") as f: json.dump(means,f,indent=2)
    np.save(np_, {c:np.array(v) for c,v in means.items()})
    logger.info(f"  Saved -> {jp}")

    with open(os.path.join(CKPT_DIR,"training_history.json"),"w") as f:
        json.dump(history,f,indent=2)

    logger.info("\n" + "="*60)
    logger.info("  ALL DONE — GACM trained on all 13 categories.")
    logger.info("  Run web server: cd web_app && python main.py")
    logger.info("="*60)

if __name__ == "__main__":
    main()
