#!/usr/bin/env python3
"""Model Performance Advisor untuk head classifier SigLIP (proyek klasifikasi sampah 3 kelas).

Membaca hasil pelatihan dan mengeluarkan:
  - diagnosa penyebab (overfit/underfit/plateau/early-stop/B3 lemah),
  - analisis potensi (headroom, trade-off Recall vs Precision B3 antar head),
  - rekomendasi terprioritas (P1/P2/P3) dengan sebab -> aksi -> tempat terapkan.

Input (salah satu / keduanya):
  --history <history_*.json>        per-epoch training history
  --comparison <perbandingan_*.csv> tabel head diurut val macro F1
  --run-dir <head_runs/RUN>         auto-discover history_*.json + perbandingan_head_*.csv

Fokus Recall B3 (B3 Precision sebagai cek false alarm). Tidak memilih model dari metrik test.
Hanya butuh pustaka standar Python.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

LABELS = ["Recylable", "Organik", "B3"]


# ----------------------------- util -----------------------------
def _last(xs):
    return xs[-1] if xs else None


def _trend(xs, window=5):
    """Arah tren beberapa titik terakhir: 'naik' | 'turun' | 'plateau' | '-'."""
    if not xs or len(xs) < 2:
        return "-"
    tail = xs[-min(window, len(xs)):]
    delta = tail[-1] - tail[0]
    span = max(abs(x) for x in tail) or 1.0
    if abs(delta) < 0.003:
        return "plateau"
    return "naik" if delta > 0 else "turun"


# ------------------------- history analysis -------------------------
def analyze_history(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        h = json.load(f)

    tr_loss = h.get("train_loss", [])
    va_loss = h.get("val_loss", [])
    tr_acc = h.get("train_acc", [])
    va_acc = h.get("val_acc", [])
    b3_rec = h.get("val_recall_B3", [])
    b3_prec = h.get("val_precision_B3", [])

    diag = {"path": os.path.basename(path), "epochs": len(va_loss), "findings": [], "flags": set()}

    if not va_loss:
        diag["findings"].append("history tidak memuat val_loss; tidak bisa dianalisis.")
        return diag

    best_epoch = int(min(range(len(va_loss)), key=lambda i: va_loss[i]))
    diag["best_epoch"] = best_epoch
    diag["best_val_loss"] = round(va_loss[best_epoch], 5)

    # overfitting: val_loss naik setelah titik terbaik + gap acc
    tail_rise = (len(va_loss) - 1 - best_epoch)
    gap_acc = (tr_acc[-1] - va_acc[-1]) if (tr_acc and va_acc) else 0.0
    if tail_rise >= 3 and _last(va_loss) > va_loss[best_epoch] + 0.01:
        diag["flags"].add("overfit")
        diag["findings"].append(
            f"val_loss naik {tail_rise} epoch setelah terbaik (best@{best_epoch}); indikasi OVERFIT.")
    if gap_acc > 0.03:
        diag["flags"].add("overfit")
        diag["findings"].append(f"gap train_acc - val_acc = {gap_acc:.3f} (>0.03); indikasi OVERFIT.")

    # underfitting: loss masih tinggi & acc rendah
    if va_acc and _last(va_acc) < 0.95 and (not tr_acc or _last(tr_acc) < 0.97):
        diag["flags"].add("underfit")
        diag["findings"].append(
            f"val_acc={_last(va_acc):.3f} relatif rendah & train juga belum tinggi; indikasi UNDERFIT.")

    # early stop terlalu dini
    if best_epoch <= 2 and len(va_loss) <= best_epoch + 9:
        diag["flags"].add("early")
        diag["findings"].append(
            f"best_epoch={best_epoch} sangat awal; cek LR/scheduler/patience (mungkin berhenti dini).")

    # B3 recall
    if b3_rec:
        diag["b3_recall_last"] = round(_last(b3_rec), 4)
        diag["b3_recall_best"] = round(max(b3_rec), 4)
        t = _trend(b3_rec)
        diag["b3_recall_trend"] = t
        if _last(b3_rec) < 0.97:
            diag["flags"].add("b3_low")
            diag["findings"].append(
                f"val_recall_B3={_last(b3_rec):.3f} < 0.97; kelas minoritas tertinggal.")
        elif t == "plateau":
            diag["findings"].append(f"val_recall_B3 plateau di ~{_last(b3_rec):.3f}.")
    if b3_prec:
        diag["b3_precision_last"] = round(_last(b3_prec), 4)
        if b3_rec and _last(b3_rec) >= 0.98 and _last(b3_prec) < 0.95:
            diag["flags"].add("b3_falsealarm")
            diag["findings"].append(
                f"B3 recall tinggi ({_last(b3_rec):.3f}) tapi B3 precision rendah "
                f"({_last(b3_prec):.3f}); model over-trigger B3 (false alarm).")

    other_rec = [h.get(f"val_recall_{c}") for c in ("Recylable", "Organik")]
    other_rec = [r for r in other_rec if r]
    if other_rec and b3_rec:
        lowest = min((_last(r) for r in other_rec), default=1.0)
        if _last(b3_rec) - lowest > 0.02 and lowest < 0.98:
            diag["findings"].append(
                f"recall kelas mayoritas ({lowest:.3f}) justru lebih rendah dari B3; "
                "error residual ada di pasangan Recylable/Organik.")

    if not diag["findings"]:
        diag["findings"].append("tidak ada bendera merah; training tampak sehat.")
    return diag


# ------------------------ comparison analysis ------------------------
def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def analyze_comparison(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"path": os.path.basename(path), "rows": [], "findings": ["CSV kosong."]}

    cols = {c.lower(): c for c in rows[0].keys()}

    def col(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        # fuzzy: cari yang mengandung
        for key, orig in cols.items():
            if any(c in key for c in cands):
                return orig
        return None

    name_c = col("model", "head", "name")
    f1_c = col("val_macro_f1", "macro_f1", "val_f1", "f1")
    b3r_c = col("val_recall_b3", "recall_b3", "b3_recall")
    b3p_c = col("val_precision_b3", "precision_b3", "b3_precision")

    enriched = []
    for r in rows:
        enriched.append({
            "name": r.get(name_c, "?") if name_c else "?",
            "macro_f1": _to_float(r.get(f1_c)) if f1_c else None,
            "b3_recall": _to_float(r.get(b3r_c)) if b3r_c else None,
            "b3_precision": _to_float(r.get(b3p_c)) if b3p_c else None,
        })

    ranked = [e for e in enriched if e["macro_f1"] is not None]
    ranked.sort(key=lambda e: -e["macro_f1"])

    out = {"path": os.path.basename(path), "rows": enriched, "findings": []}
    if ranked:
        best, worst = ranked[0], ranked[-1]
        out["best"], out["worst"] = best, worst
        spread = best["macro_f1"] - worst["macro_f1"]
        out["spread"] = spread
        out["findings"].append(
            f"head terbaik '{best['name']}' val macro F1={best['macro_f1']:.4f}; "
            f"terburuk '{worst['name']}'={worst['macro_f1']:.4f} (selisih {spread:.4f}).")
        if spread < 0.01:
            out["findings"].append(
                "semua head rapat (<0.01); headroom arsitektur kecil -> alihkan effort ke DATA & ensemble.")

        # trade-off B3 recall vs precision
        with_b3 = [e for e in enriched if e["b3_recall"] is not None and e["b3_precision"] is not None]
        if with_b3:
            best_b3r = max(with_b3, key=lambda e: e["b3_recall"])
            out["findings"].append(
                f"Recall B3 tertinggi: '{best_b3r['name']}' = {best_b3r['b3_recall']:.4f} "
                f"(B3 precision {best_b3r['b3_precision']:.4f}).")
            risky = [e for e in with_b3 if e["b3_recall"] >= 0.98 and e["b3_precision"] < 0.95]
            if risky:
                out["findings"].append(
                    "ada head dengan B3 recall tinggi tapi precision < 0.95 (false alarm): "
                    + ", ".join(e["name"] for e in risky) + ".")
    return out


# --------------------------- rekomendasi ---------------------------
def build_recommendations(hist_list, comp, b3_target):
    flags = set()
    for d in hist_list:
        flags |= d.get("flags", set())

    recs = []  # (prioritas, sebab, aksi, tempat)

    if "b3_low" in flags:
        recs.append(("P1", "Recall B3 di bawah target (kelas minoritas tertinggal)",
                     "Tambah augmentasi B3 (naikkan N_AUG B3), naikkan bobot kelas B3 di focal loss, "
                     "atau turunkan threshold keputusan B3; pantau B3 Precision.",
                     "N_AUG & class weights (Cell config) + threshold-tuning (Cell evaluasi)"))
    if "b3_falsealarm" in flags:
        recs.append(("P1", "B3 over-trigger: recall tinggi tapi precision rendah",
                     "Kurangi bobot/threshold B3, dan bersihkan near-duplicate/label noise yang "
                     "mengaburkan batas kelas B3.",
                     "class weights/threshold + skill embedding-data-diagnostics"))
    if "overfit" in flags:
        recs.append(("P1", "Overfitting (val_loss naik / gap acc besar)",
                     "Naikkan DROPOUT_RATE & HEAD_WEIGHT_DECAY, pakai head lebih sederhana "
                     "(linear/mlp), tambah augmentasi; biarkan early stopping bekerja.",
                     "Cell config head di siglip_embedding_probe_colab.ipynb"))
    if "underfit" in flags:
        recs.append(("P2", "Underfitting (loss tinggi, acc belum jenuh)",
                     "Pakai head lebih kuat (mlp2/resmlp), latih lebih lama / naikkan patience, "
                     "kurangi regularisasi, pastikan INPUT_NORM='standardize'.",
                     "Cell config head"))
    if "early" in flags:
        recs.append(("P2", "Early stop terlalu dini",
                     "Turunkan LR atau pakai scheduler lebih landai, naikkan EARLY_STOP_PATIENCE; "
                     "cek apakah val_loss noisy.",
                     "Cell config head"))

    # dari comparison
    if comp:
        spread = comp.get("spread")
        if spread is not None and spread < 0.01:
            recs.append(("P1", "Semua head rapat (~99%), headroom arsitektur kecil",
                         "Fokus ke DATA: jalankan skill embedding-data-diagnostics untuk cari label "
                         "noise, near-duplicate lintas split, dan sampel ambigu; perbaiki di "
                         "preprocessing_lokal.ipynb lalu re-ekstrak embedding.",
                         "skill embedding-data-diagnostics + preprocessing_lokal.ipynb"))
            recs.append(("P2", "Head individual sudah mentok",
                         "Coba ensemble (multi-seed head terbaik / gabungan head) dan threshold-tuning "
                         "untuk menaikkan Recall B3 beberapa poin.",
                         "Cell ensemble & evaluasi"))
        best = comp.get("best")
        if best:
            recs.append(("P2", f"Head terbaik saat ini '{best['name']}' (val macro F1 {best['macro_f1']:.4f})",
                         "Jadikan baseline; evaluasi test SEKALI hanya untuk model final terpilih.",
                         "report_test('nama_model') di notebook"))

    # selalu ada: data-centric sebagai jalur umum bila error residual pasangan kelas
    if any("Recylable/Organik" in fi for d in hist_list for fi in d.get("findings", [])):
        recs.append(("P2", "Error residual terpusat di pasangan Recylable<->Organik",
                     "Diagnosa & bersihkan pasangan itu (relabel ambigu, pisah sub-kategori) via "
                     "embedding-data-diagnostics, bukan tuning global.",
                     "skill embedding-data-diagnostics"))

    if not recs:
        recs.append(("P3", "Tidak ada bendera merah terdeteksi",
                     "Pertahankan konfig; untuk gain lebih lanjut, jalankan embedding-data-diagnostics "
                     "dan coba ensemble + threshold-tuning.",
                     "skill embedding-data-diagnostics + Cell ensemble"))

    order = {"P1": 0, "P2": 1, "P3": 2}
    recs.sort(key=lambda r: order[r[0]])
    return recs


# ------------------------------ main ------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", action="append", default=[],
                    help="path history_*.json (boleh diulang)")
    ap.add_argument("--comparison", help="path perbandingan_head_*.csv")
    ap.add_argument("--run-dir", help="folder head_runs/<RUN> (auto-discover)")
    ap.add_argument("--b3-recall-target", type=float, default=0.99)
    ap.add_argument("--out", help="tulis rekomendasi ke file markdown")
    args = ap.parse_args()

    histories = list(args.history)
    comparison = args.comparison

    if args.run_dir:
        histories += sorted(glob.glob(os.path.join(args.run_dir, "history_*.json")))
        if not comparison:
            comps = sorted(glob.glob(os.path.join(args.run_dir, "perbandingan_head_*.csv")))
            comparison = comps[0] if comps else None

    histories = list(dict.fromkeys(histories))  # dedup, jaga urutan
    if not histories and not comparison:
        print("[FAIL] Beri --history, --comparison, atau --run-dir.")
        return 1

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("================ MODEL PERFORMANCE ADVISOR ================")

    hist_list = []
    if histories:
        out("\n--- DIAGNOSA HISTORY TRAINING ---")
        for hp in histories:
            try:
                d = analyze_history(hp)
            except Exception as e:
                out(f"[WARN] gagal baca {hp}: {e}")
                continue
            hist_list.append(d)
            out(f"\n[{d['path']}] epochs={d.get('epochs','?')} "
                f"best_epoch={d.get('best_epoch','?')} best_val_loss={d.get('best_val_loss','?')}")
            if "b3_recall_last" in d:
                out(f"  Recall B3 terakhir={d['b3_recall_last']} (best {d.get('b3_recall_best')}, "
                    f"tren {d.get('b3_recall_trend')}) | B3 precision terakhir={d.get('b3_precision_last','?')}")
            for fi in d["findings"]:
                out(f"  - {fi}")

    comp = None
    if comparison:
        out("\n--- ANALISIS POTENSI (PERBANDINGAN HEAD) ---")
        try:
            comp = analyze_comparison(comparison)
            out(f"[{comp['path']}]")
            for fi in comp["findings"]:
                out(f"  - {fi}")
        except Exception as e:
            out(f"[WARN] gagal baca comparison: {e}")
            comp = None

    out("\n--- REKOMENDASI TERPRIORITAS ---")
    recs = build_recommendations(hist_list, comp, args.b3_recall_target)
    for i, (prio, sebab, aksi, tempat) in enumerate(recs, 1):
        out(f"\n{prio} #{i}  SEBAB : {sebab}")
        out(f"        AKSI  : {aksi}")
        out(f"        DI    : {tempat}")

    out("\nCatatan: tiap perubahan konfig -> ganti RUN_NAME; bandingkan val sebelum/sesudah. "
        "Test hanya disentuh sekali untuk model final.")

    if args.out:
        Path(args.out).write_text("\n".join("    " + l if l.startswith(("P1","P2","P3"," ")) else l
                                             for l in lines), encoding="utf-8")
        print(f"\n[OK] Rekomendasi ditulis ke {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
