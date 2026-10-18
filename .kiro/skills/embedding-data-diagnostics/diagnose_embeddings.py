#!/usr/bin/env python3
"""Diagnosa data-centric di atas embedding SigLIP (.npz).

Menemukan data yang kemungkinan MENURUNKAN performa model klasifikasi sampah 3 kelas:
  - label noise (label beda dari mayoritas tetangga + diprediksi salah dgn yakin),
  - near-duplicate lintas split (potensi kebocoran metrik),
  - sampel ambigu (margin kecil, perbatasan kelas),
  - outlier terhadap centroid kelas,
  - ringkasan pasangan kelas yang paling sering tertukar.

Bekerja offline terhadap embeddings_{train,val,test}.npz yang sudah diunduh dari Drive.
Hanya butuh numpy (scikit-learn opsional, dipakai bila tersedia untuk kNN lebih cepat).

Kontrak .npz: keys embeddings[M,1152], labels[M], label_names[M], paths[M],
orig_idx[M], view_idx[M]. Label index {Recylable:0, Organik:1, B3:2}.

Semua analisis per-gambar memakai view_idx == 0. Statistik standardisasi dari TRAIN saja.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

LABELS = ["Recylable", "Organik", "B3"]  # index 0,1,2 ; ejaan 'Recylable' disengaja


def load_split(embedding_dir: str, split: str) -> dict | None:
    path = os.path.join(embedding_dir, f"embeddings_{split}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def view0(d: dict) -> dict:
    """Ambil hanya view asli (view_idx == 0); fallback ke dedup orig_idx bila tak ada."""
    if "view_idx" in d:
        m = d["view_idx"] == 0
    else:
        m = np.ones(len(d["embeddings"]), dtype=bool)
    out = {k: (v[m] if hasattr(v, "__len__") and len(v) == len(m) else v) for k, v in d.items()}
    return out


def l2(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def label_name_arr(d: dict) -> np.ndarray:
    if "label_names" in d:
        return d["label_names"].astype(str)
    return np.array([LABELS[int(i)] for i in d["labels"]])


def knn_indices(query: np.ndarray, base: np.ndarray, k: int, exclude_self: bool):
    """Index k tetangga terdekat (cosine) tiap baris query terhadap base.
    query & base harus sudah L2-normalized. Mengembalikan (idx[k], sim[k])."""
    try:
        from sklearn.neighbors import NearestNeighbors
        n = k + (1 if exclude_self else 0)
        nn = NearestNeighbors(n_neighbors=min(n, len(base)), metric="cosine")
        nn.fit(base)
        dist, idx = nn.kneighbors(query)
        sim = 1.0 - dist
        if exclude_self:
            idx, sim = idx[:, 1:], sim[:, 1:]
        return idx, sim
    except Exception:
        # fallback numpy (O(N*Q*D)); ok untuk beberapa ribu sampel
        sims = query @ base.T
        if exclude_self:
            np.fill_diagonal(sims, -np.inf)
        idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        row = np.arange(len(query))[:, None]
        order = np.argsort(-sims[row, idx], axis=1)
        idx = idx[row, order]
        sim = sims[row, idx]
        return idx, sim


def write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding-dir", required=True,
                    help="folder berisi embeddings_train.npz|val.npz|test.npz")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--dup-threshold", type=float, default=0.97)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out-dir", default="Output_data_diagnostics")
    args = ap.parse_args()

    train = load_split(args.embedding_dir, "train")
    if train is None:
        print(f"[FAIL] embeddings_train.npz tidak ada di {args.embedding_dir}")
        return 1
    val = load_split(args.embedding_dir, "val")
    test = load_split(args.embedding_dir, "test")

    out = Path(args.out_dir)

    # view 0 saja untuk analisis per-gambar
    tr = view0(train)
    Xtr_raw = tr["embeddings"].astype(np.float32)
    ytr = tr["labels"].astype(int)
    ytr_name = label_name_arr(tr)
    ptr = tr["paths"].astype(str)

    # standardisasi (statistik dari train saja), lalu L2 untuk cosine
    mu = Xtr_raw.mean(0)
    sd = Xtr_raw.std(0) + 1e-6
    Xtr = l2((Xtr_raw - mu) / sd)

    print(f"[INFO] train view0: {len(Xtr)} gambar | dim {Xtr.shape[1]} | k={args.k}")
    dist = {lab: int((ytr_name == lab).sum()) for lab in LABELS}
    print(f"[INFO] distribusi train: {dist}")

    # ---- 1. kNN leave-one-out di train: label noise + confusion summary ----
    idx, sim = knn_indices(Xtr, Xtr, k=args.k, exclude_self=True)
    neigh_lab = ytr[idx]  # [N,k]

    # suara tetangga per kelas (vote x bobot cosine), pakai bobot kelas inverse-freq ringan
    counts = np.array([dist[l] for l in LABELS], dtype=np.float32)
    cls_w = (counts.sum() / (3 * np.clip(counts, 1, None))).astype(np.float32)
    votes = np.zeros((len(Xtr), 3), dtype=np.float32)
    for c in range(3):
        mask = (neigh_lab == c).astype(np.float32)
        votes[:, c] = (mask * np.clip(sim, 0, None)).sum(1) * cls_w[c]
    pred = votes.argmax(1)
    vsum = votes.sum(1, keepdims=True)
    prob = votes / np.clip(vsum, 1e-9, None)
    conf = prob.max(1)
    margin = np.sort(prob, 1)[:, -1] - np.sort(prob, 1)[:, -2]

    # label noise: diprediksi salah, confidence tinggi, mayoritas tetangga tak setuju label asli
    frac_same = (neigh_lab == ytr[:, None]).mean(1)
    wrong = pred != ytr
    noise_score = conf * (1.0 - frac_same)
    noise_rank = np.where(wrong, noise_score, 0.0)
    order = np.argsort(-noise_rank)
    noise_rows = []
    for i in order:
        if noise_rank[i] <= 0:
            break
        noise_rows.append([
            ytr_name[i], LABELS[pred[i]], f"{conf[i]:.3f}", f"{frac_same[i]:.3f}",
            f"{noise_rank[i]:.4f}", ptr[i],
        ])
        if len(noise_rows) >= args.top:
            break
    write_csv(out / "label_noise_candidates.csv",
              ["label_asli", "pred_knn", "conf", "frac_tetangga_sama", "noise_score", "path"],
              noise_rows)

    # confusion summary (leave-one-out train)
    conf_rows = []
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            n = int(((ytr == a) & (pred == b)).sum())
            if n:
                conf_rows.append([LABELS[a], LABELS[b], n])
    conf_rows.sort(key=lambda r: -r[2])
    write_csv(out / "confusion_summary.csv", ["label_asli", "pred_knn", "jumlah"], conf_rows)

    # ---- 2. sampel ambigu: margin probabilitas kecil ----
    amb_order = np.argsort(margin)
    amb_rows = [[ytr_name[i], LABELS[pred[i]], f"{margin[i]:.4f}", f"{conf[i]:.3f}", ptr[i]]
                for i in amb_order[: args.top]]
    write_csv(out / "ambiguous_samples.csv",
              ["label_asli", "pred_knn", "margin", "conf", "path"], amb_rows)

    # ---- 3. outlier terhadap centroid kelas ----
    out_rows = []
    for c in range(3):
        m = ytr == c
        if m.sum() == 0:
            continue
        centroid = l2(Xtr[m].mean(0, keepdims=True))[0]
        sim_c = Xtr[m] @ centroid
        sub_idx = np.where(m)[0]
        worst = sub_idx[np.argsort(sim_c)][: max(1, args.top // 3)]
        for i in worst:
            out_rows.append([ytr_name[i], f"{float(Xtr[i] @ centroid):.3f}", ptr[i]])
    out_rows.sort(key=lambda r: float(r[1]))
    write_csv(out / "class_centroid_outliers.csv",
              ["label", "cosine_ke_centroid", "path"], out_rows)

    # ---- 4. near-duplicate lintas split (train vs val/test) ----
    dup_rows = []
    for split_name, dsplit in (("val", val), ("test", test)):
        if dsplit is None:
            continue
        ds = view0(dsplit)
        Xs = l2((ds["embeddings"].astype(np.float32) - mu) / sd)
        ps = ds["paths"].astype(str)
        ls = label_name_arr(ds)
        nidx, nsim = knn_indices(Xs, Xtr, k=1, exclude_self=False)
        for j in range(len(Xs)):
            s = float(nsim[j, 0])
            if s >= args.dup_threshold:
                t = int(nidx[j, 0])
                dup_rows.append([split_name, ls[j], ps[j], ytr_name[t], ptr[t], f"{s:.4f}"])
    dup_rows.sort(key=lambda r: -float(r[5]))
    write_csv(out / "cross_split_duplicates.csv",
              ["split", "label_split", "path_split", "label_train", "path_train_match", "cosine"],
              dup_rows[: max(args.top, 200)])

    # ---- ringkasan ----
    print("\n================ RINGKASAN DIAGNOSTIK ================")
    print(f"Kandidat label noise (train)      : {len(noise_rows)} (top ditulis)")
    noise_b3 = sum(1 for r in noise_rows if r[0] == "B3")
    print(f"  - di antaranya label asli B3     : {noise_b3}  <- prioritas, cek dulu")
    print(f"Sampel ambigu (margin kecil)       : {len(amb_rows)} ditulis")
    print(f"Outlier centroid kelas             : {len(out_rows)} ditulis")
    print(f"Near-duplicate lintas split (>= {args.dup_threshold}): {len(dup_rows)}  <- potensi kebocoran metrik")
    if conf_rows:
        top_conf = conf_rows[0]
        print(f"Pasangan paling tertukar (kNN LOO) : {top_conf[0]} -> {top_conf[1]} ({top_conf[2]}x)")
    print(f"\nCSV lengkap di: {out.resolve()}")
    print("Tindak lanjut: tinjau manual -> perbaiki di preprocessing_lokal.ipynb -> re-ekstrak embedding ke dir baru.")
    print("Jangan auto-apply relabel; keputusan label butuh verifikasi manusia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
