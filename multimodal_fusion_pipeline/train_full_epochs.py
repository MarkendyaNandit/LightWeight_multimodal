"""
train_full_epochs.py — Multi-Epoch GACM Fine-Tuning Across All 13 Categories.

Categories (13):
  MVTec 3D: bagel, cable_gland, carrot, cookie, dowel, foam, peach, potato, rope, tire
  Hardware:  phone_screen (MSD), car_metal (NEU-DET), pcb (DeepPCB)

Training Strategy:
  NT-Xent Contrastive Loss (normal vs augmented-defect pairs)
  Augmentation: random crop, flip, jitter, rotation, grayscale, gaussian blur
  GACM fine-tuned for NUM_EPOCHS=30 epochs, early stopping patience=5
  AdamW optimizer + cosine annealing LR scheduler
  After all epochs: recompute per-category manifold centers and save to checkpoints

Outputs:
  outputs/checkpoints/all_category_means.json
  outputs/checkpoints/all_category_means.npy
  outputs/checkpoints/best_fusion_model_finetuned.pth
  outputs/checkpoints/training_history.json
"""

import os, sys, json, logging, time, random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

# ─── Config ──────────────────────────────────────────────
NUM_EPOCHS   = 30
BATCH_SIZE   = 16
LR           = 1e-4
PATIENCE     = 5
TEMP         = 0.07
WEIGHT_DECAY = 1e-4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train-Full-Epochs")

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
depth_mod    = load_module("depth_mod", os.path.join(DEPTH_DIR, "models", "depth_encoder.py"))
DepthEncoder = depth_mod.DepthEncoder
gacm_mod        = load_module("gacm_mod", os.path.join(CURRENT_DIR, "gacm.py"))
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

def get_normal_files(cat: str) -> List[str]:
    if cat in ["bagel","cable_gland","carrot","cookie","dowel","foam","peach","potato","rope","tire"]:
        return _scan(os.path.join(DATASET_ROOT, cat, "train", "good", "rgb"))
    elif cat == "phone_screen":
        # MSD only has 20 "good" images. Augment normals by treating all MSD images
        # as phone screen textures (defects are local; global manifold is still phone-screen)
        files = _scan(os.path.join(BASE_DIR, "archive", "good"))
        for sub in ["scratch", "oil", "stain"]:
            files += _scan(os.path.join(BASE_DIR, "archive", sub))
        return files[:300]  # cap at 300
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

def get_defect_files(cat: str) -> List[str]:
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

# ─── Transforms ───────────────────────────────────────────
NORM = dict(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
base_transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(**NORM)])
aug_transform  = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7,1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.1),
    T.GaussianBlur(kernel_size=5, sigma=(0.1,2.0)),
    T.ToTensor(),
    T.Normalize(**NORM)
])

# ─── Dataset ──────────────────────────────────────────────
class AnomalyContrastDataset(Dataset):
    def __init__(self, cat_normal_map, cat_defect_map):
        self.samples    = []
        self.cat_to_id  = {c:i for i,c in enumerate(CATEGORIES)}
        for cat, normal_files in cat_normal_map.items():
            defect_files = cat_defect_map.get(cat, [])
            for nf in normal_files:
                df = random.choice(defect_files) if defect_files else nf
                self.samples.append((nf, df, self.cat_to_id[cat]))
        random.shuffle(self.samples)

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        np_, dp_, cat_id = self.samples[idx]
        def safe_open(p):
            try:    return Image.open(p).convert("RGB")
            except: return Image.new("RGB",(224,224))
        nimg = safe_open(np_)
        dimg = safe_open(dp_)
        return (
            base_transform(nimg),
            base_transform(nimg.convert("L").convert("RGB")),
            aug_transform(nimg),
            aug_transform(dimg),
            cat_id
        )

# ─── Losses ───────────────────────────────────────────────
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature
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

# ─── Load Models ──────────────────────────────────────────
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

def load_models():
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

    fusion_model = VisualFusionPipeline().to(DEVICE)
    ck = os.path.join(CKPT_DIR, "best_fusion_model.pth")
    if os.path.isfile(ck):
        fusion_model.load_state_dict(extract_sd(torch.load(ck,map_location=DEVICE,weights_only=False)))
        logger.info("  [LOADED] GACM (prior checkpoint)")
    else:
        logger.info("  [NEW] GACM — training from scratch")

    # Freeze encoders; only train GACM
    for p in rgb_model.parameters():   p.requires_grad = False
    for p in depth_model.parameters(): p.requires_grad = False
    rgb_model.eval(); depth_model.eval()
    return rgb_model, depth_model, fusion_model

# ─── Compute Manifold Means ───────────────────────────────
@torch.no_grad()
def compute_category_means(cat_normal_map, rgb_model, depth_model, fusion_model):
    fusion_model.eval(); rgb_model.eval(); depth_model.eval()
    out = {}
    for cat, files in cat_normal_map.items():
        if not files: continue
        vecs = []
        for i in range(0, len(files), 32):
            batch = files[i:i+32]
            ri, di = [], []
            for p in batch:
                try:    img = Image.open(p).convert("RGB")
                except: img = Image.new("RGB",(224,224))
                ri.append(base_transform(img).unsqueeze(0))
                di.append(base_transform(img.convert("L").convert("RGB")).unsqueeze(0))
            fr = rgb_model(torch.cat(ri,0).to(DEVICE))
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
    logger.info("  FULL MULTI-EPOCH GACM FINE-TUNING  (13 Categories)")
    logger.info(f"  Epochs={NUM_EPOCHS}  BatchSize={BATCH_SIZE}  LR={LR}  Patience={PATIENCE}")
    logger.info("="*60)

    # 1. File lists
    logger.info("\n[1/5] Scanning datasets...")
    cat_normal_map, cat_defect_map = {}, {}
    for cat in CATEGORIES:
        n = get_normal_files(cat); d = get_defect_files(cat)
        cat_normal_map[cat] = n; cat_defect_map[cat] = d
        logger.info(f"  {cat:15s}  normal={len(n):4d}  defect={len(d):4d}")

    # 2. Dataset
    logger.info("\n[2/5] Building contrastive dataset...")
    ds = AnomalyContrastDataset(cat_normal_map, cat_defect_map)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, pin_memory=(DEVICE=="cuda"), drop_last=True)
    logger.info(f"  Total samples: {len(ds)}  |  Steps/epoch: {len(dl)}")

    # 3. Models
    logger.info("\n[3/5] Loading models...")
    rgb_model, depth_model, fusion_model = load_models()

    # 4. Optimizer/Scheduler/Loss
    optimizer    = torch.optim.AdamW(fusion_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    ntxent_loss  = NTXentLoss(temperature=TEMP)
    center_loss  = CatCenterLoss(len(CATEGORIES)).to(DEVICE)
    center_optim = torch.optim.Adam(center_loss.parameters(), lr=1e-3)

    best_loss = float("inf"); patience_ct = 0; history = []

    logger.info(f"\n[4/5] Training GACM for up to {NUM_EPOCHS} epochs...")
    logger.info("="*60)

    for epoch in range(1, NUM_EPOCHS+1):
        fusion_model.train()
        e_loss = e_ntx = e_ctr = 0.0
        t0 = time.time()

        for anc_rgb, anc_dep, pos_rgb, neg_rgb, cat_ids in dl:
            anc_rgb = anc_rgb.to(DEVICE); anc_dep = anc_dep.to(DEVICE)
            pos_rgb = pos_rgb.to(DEVICE); neg_rgb = neg_rgb.to(DEVICE)
            cat_ids = cat_ids.to(DEVICE)

            with torch.no_grad():
                fr_a = rgb_model(anc_rgb);  fd_a,_ = depth_model(anc_dep)
                fr_p = rgb_model(pos_rgb);  fd_p,_ = depth_model(pos_rgb.mean(1,keepdim=True).expand_as(pos_rgb))
                fr_n = rgb_model(neg_rgb);  fd_n,_ = depth_model(neg_rgb.mean(1,keepdim=True).expand_as(neg_rgb))

            z_a = fusion_model(fr_a, fd_a)
            z_p = fusion_model(fr_p, fd_p)
            z_n = fusion_model(fr_n, fd_n)

            l_ntx = ntxent_loss(z_a, z_p, z_n)
            l_ctr = center_loss(z_a, cat_ids)
            loss  = l_ntx + 0.3*l_ctr

            optimizer.zero_grad(); center_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step(); center_optim.step()

            e_loss += loss.item(); e_ntx += l_ntx.item(); e_ctr += l_ctr.item()

        scheduler.step()
        n  = max(len(dl),1)
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
            sp = os.path.join(CKPT_DIR,"best_fusion_model_finetuned.pth")
            torch.save(fusion_model.state_dict(), sp)
            logger.info(f"  ✓ Saved best model → {sp}")
        else:
            patience_ct += 1
            logger.info(f"  No improvement ({patience_ct}/{PATIENCE})")
            if patience_ct >= PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch}.")
                break

    logger.info(f"\nTraining done. Best loss={best_loss:.4f}")

    # 5. Final manifold means
    logger.info("\n[5/5] Recomputing manifold centers with best weights...")
    bp = os.path.join(CKPT_DIR,"best_fusion_model_finetuned.pth")
    if os.path.isfile(bp):
        fusion_model.load_state_dict(torch.load(bp,map_location=DEVICE,weights_only=False))
        logger.info("  Best weights loaded.")

    means = compute_category_means(cat_normal_map, rgb_model, depth_model, fusion_model)
    jp = os.path.join(CKPT_DIR,"all_category_means.json")
    np_ = os.path.join(CKPT_DIR,"all_category_means.npy")
    with open(jp,"w") as f: json.dump(means,f,indent=2)
    np.save(np_, {c:np.array(v) for c,v in means.items()})
    logger.info(f"  Saved manifold centers → {jp}")

    hp = os.path.join(CKPT_DIR,"training_history.json")
    with open(hp,"w") as f: json.dump(history,f,indent=2)
    logger.info(f"  Saved training history → {hp}")

    logger.info("\n" + "="*60)
    logger.info("  ALL DONE — Model ready for inference.")
    logger.info("="*60)

if __name__ == "__main__":
    main()
