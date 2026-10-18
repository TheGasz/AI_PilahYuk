import io
import os
import glob
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import SiglipVisionModel, SiglipImageProcessor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# 1. DEFINISI ARSITEKTUR HEAD (Persis Notebook Cell 26)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_act(name: str) -> nn.Module:
    return nn.SiLU() if name == "silu" else nn.GELU()


def build_head(head_type: str, in_dim: int, num_classes: int, dropout: float = 0.3, act: str = "gelu") -> nn.Module:
    if head_type == "linear":
        return nn.Linear(in_dim, num_classes)
    elif head_type == "mlp":
        h = in_dim // 2
        return nn.Sequential(nn.Linear(in_dim, h), make_act(act), nn.Dropout(dropout), nn.Linear(h, num_classes))
    elif head_type == "mlp_norm":
        h = in_dim // 2
        return nn.Sequential(nn.Linear(in_dim, h), RMSNorm(h), make_act(act), nn.Dropout(dropout), nn.Linear(h, num_classes))
    elif head_type == "mlp2":
        return nn.Sequential(
            nn.Linear(in_dim, 512), RMSNorm(512), make_act(act), nn.Dropout(dropout),
            nn.Linear(512, 128), RMSNorm(128), make_act(act), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    raise ValueError(f"Unsupported head_type: {head_type}")


def _load_ckpt(ckpt_path: str, device: torch.device):
    """Muat satu checkpoint head: (model eval, norm_dict, idx_to_label)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_head(
        ckpt["head_type"], ckpt["in_dim"], ckpt["num_classes"],
        ckpt.get("dropout", 0.3), ckpt.get("act", "gelu"),
    ).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    raw_norm = ckpt.get("input_norm", {}) or {}
    norm = {
        "mode": raw_norm.get("mode", "standardize"),
        "mu": raw_norm["mu"].to(device, dtype=torch.float32) if "mu" in raw_norm else None,
        "sd": raw_norm["sd"].to(device, dtype=torch.float32) if "sd" in raw_norm else None,
    }
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {int(v): str(k) for k, v in label_to_idx.items()}
    return model, norm, idx_to_label


def _apply_norm(emb: torch.Tensor, norm: Dict[str, Any]) -> torch.Tensor:
    x = emb.float()
    if norm["mode"] == "l2":
        return x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-12)
    if norm["mode"] == "standardize" and norm["mu"] is not None and norm["sd"] is not None:
        return (x - norm["mu"]) / norm["sd"]
    return x


def _format_probs(probs: np.ndarray, idx_to_label: Dict[int, str]) -> Dict[str, Any]:
    top_idx = int(np.argmax(probs))
    return {
        "label": idx_to_label[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "probabilities": {idx_to_label[i]: round(float(probs[i]), 4) for i in range(len(probs))},
    }


class LoadedHead:
    """Head tunggal (dipakai untuk Sub-Head per Kategori)."""
    def __init__(self, ckpt_path: str, device: torch.device):
        self.model, self.norm, self.idx_to_label = _load_ckpt(ckpt_path, device)

    @torch.inference_mode()
    def predict(self, emb: torch.Tensor) -> Dict[str, Any]:
        logits = self.model(_apply_norm(emb, self.norm))
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return _format_probs(probs, self.idx_to_label)


class EnsembleHead:
    """Ensemble rata-rata probabilitas beberapa head seed (ens_seed_mlp2).
    Prediksi = mean(softmax(logits_i)) lalu argmax. Tiap head memakai input_norm
    checkpoint-nya sendiri. Semua head WAJIB punya idx_to_label identik."""
    def __init__(self, ckpt_paths: List[str], device: torch.device):
        if not ckpt_paths:
            raise RuntimeError("EnsembleHead: daftar checkpoint kosong.")
        self.members = []
        ref_map = None
        for p in ckpt_paths:
            model, norm, idx_to_label = _load_ckpt(p, device)
            if ref_map is None:
                ref_map = idx_to_label
            elif idx_to_label != ref_map:
                raise RuntimeError(
                    f"EnsembleHead: label_to_idx tidak konsisten antar seed.\n"
                    f"  {os.path.basename(p)}: {idx_to_label}\n  acuan: {ref_map}")
            self.members.append((model, norm))
        self.idx_to_label = ref_map
        self.n = len(self.members)

    @torch.inference_mode()
    def predict(self, emb: torch.Tensor) -> Dict[str, Any]:
        acc = None
        for model, norm in self.members:
            logits = model(_apply_norm(emb, norm))
            p = torch.softmax(logits, dim=-1)
            acc = p if acc is None else acc + p
        probs = (acc / self.n).squeeze(0).cpu().numpy()
        return _format_probs(probs, self.idx_to_label)


# ============================================================
# 2. STATE GLOBAL & LIFESPAN STARTUP
# ============================================================
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "google/siglip-so400m-patch14-384")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
IMAGE_SIZE = 384
MAX_IMAGE_SIDE = 1024

# Pola file 5 head mlp2 multi-seed (ens_seed_mlp2). Seed 42 = file tanpa suffix _sNN.
ENSEMBLE_GLOB = "best_head_*mlp2.pt"          # cocok: *_mlp2.pt (seed 42)
ENSEMBLE_SEED_GLOB = "best_head_*mlp2_s*.pt"  # cocok: *_mlp2_s43.pt ... _s46.pt

# Sub-Head per Kategori. Kunci = label yang dikeluarkan Main Head (ejaan `Recylable`).
SUB_HEAD_FILES = {
    "Recylable": "subRecyclable.pt",
    "Organik": "subOrganik.pt",
    "B3": "subB3.pt",
}

ml_registry: Dict[str, Any] = {}


def _resolve_ensemble_paths() -> List[str]:
    """Kumpulkan path 5 head mlp2 (seed 42 + s43..s46), unik & terurut."""
    base = set(glob.glob(os.path.join(ARTIFACTS_DIR, ENSEMBLE_GLOB)))
    seeds = set(glob.glob(os.path.join(ARTIFACTS_DIR, ENSEMBLE_SEED_GLOB)))
    paths = sorted(base | seeds)
    # Buang file lain yang kebetulan tertangkap (mis. mainHead.pt tidak cocok pola).
    return paths


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[STARTUP] Memuat SigLIP backbone di device={device}, dtype={dtype}...")
    proc = SiglipImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    vision_model = SiglipVisionModel.from_pretrained(MODEL_CHECKPOINT).to(device=device, dtype=dtype).eval()

    preprocess_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
    ])

    # Main Head = ens_seed_mlp2 (rata-rata 5 head mlp2 multi-seed)
    ens_paths = _resolve_ensemble_paths()
    if not ens_paths:
        raise RuntimeError(
            f"Tidak menemukan checkpoint ensemble di {ARTIFACTS_DIR} "
            f"(pola {ENSEMBLE_GLOB} / {ENSEMBLE_SEED_GLOB}).")
    print(f"[STARTUP] Main Head ens_seed_mlp2 dari {len(ens_paths)} seed:")
    for p in ens_paths:
        print(f"           - {os.path.basename(p)}")
    main_head = EnsembleHead(ens_paths, device)

    # Sub-Heads per Kategori
    sub_heads = {}
    for kat, fname in SUB_HEAD_FILES.items():
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        if not os.path.exists(fpath):
            raise RuntimeError(f"Sub-head {kat} tidak ditemukan: {fpath}")
        sub_heads[kat] = LoadedHead(fpath, device)

    # Sanity: label Main Head harus sama dengan kunci routing Sub-Head
    main_labels = set(main_head.idx_to_label.values())
    if main_labels != set(sub_heads.keys()):
        print(f"[WARN] label Main Head {main_labels} != kunci Sub-Head {set(sub_heads.keys())}. "
              f"Routing hierarkis bisa gagal untuk kategori yang tak cocok.")

    ml_registry.update({
        "device": device, "dtype": dtype,
        "vision_model": vision_model, "preprocess_tf": preprocess_tf,
        "main_head": main_head, "sub_heads": sub_heads,
    })
    print(f"[STARTUP] Siap. Main=ens_seed_mlp2({main_head.n} seed), "
          f"Sub={list(sub_heads.keys())}.")
    yield
    ml_registry.clear()


app = FastAPI(title="SigLIP Waste Classification API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan dengan domain web app Anda di produksi
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    main_label: str
    main_confidence: float
    main_probabilities: Dict[str, float]
    sub_label: str
    sub_confidence: float
    sub_probabilities: Dict[str, float]
    inference_time_ms: float
    device_used: str


@app.get("/health")
def health_check():
    mh = ml_registry.get("main_head")
    return {
        "status": "healthy" if mh is not None else "uninitialized",
        "device": str(ml_registry.get("device", "uninitialized")),
        "ensemble_seeds": getattr(mh, "n", 0),
        "sub_heads": list(ml_registry.get("sub_heads", {}).keys()),
    }


# `def` (bukan `async def`) -> FastAPI menjalankannya di threadpool, komputasi PyTorch
# tidak memblokir event loop saat menerima request bersamaan.
@app.post("/predict", response_model=PredictionResponse)
def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")

    t0 = time.perf_counter()
    try:
        raw_bytes = file.file.read()
        img = Image.open(io.BytesIO(raw_bytes))
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        img = img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gambar tidak valid: {e}")

    device = ml_registry["device"]
    dtype = ml_registry["dtype"]
    vision_model = ml_registry["vision_model"]
    preprocess_tf = ml_registry["preprocess_tf"]

    # 1. Ekstraksi embedding sekali dari SigLIP backbone
    with torch.inference_mode():
        pixel_values = preprocess_tf(img).unsqueeze(0).to(device=device, dtype=dtype)
        outputs = vision_model(pixel_values=pixel_values)
        emb = outputs.pooler_output.float()  # [1, 1152]

    # 2. Prediksi kategori utama (ens_seed_mlp2): Recylable / Organik / B3
    main_res = ml_registry["main_head"].predict(emb)
    pred_kat = main_res["label"]

    # 3. Routing hierarkis ke Sub-Head sesuai kategori utama
    sub_head = ml_registry["sub_heads"].get(pred_kat)
    if sub_head is None:
        raise HTTPException(status_code=500, detail=f"Sub-head untuk kategori {pred_kat} tidak ditemukan.")
    sub_res = sub_head.predict(emb)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return PredictionResponse(
        main_label=main_res["label"],
        main_confidence=main_res["confidence"],
        main_probabilities=main_res["probabilities"],
        sub_label=sub_res["label"],
        sub_confidence=sub_res["confidence"],
        sub_probabilities=sub_res["probabilities"],
        inference_time_ms=round(elapsed_ms, 2),
        device_used=str(device),
    )
