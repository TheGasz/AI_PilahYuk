#!/usr/bin/env python3
"""Methodology Fault Fixer untuk pipeline klasifikasi sampah SigLIP.

Mendeteksi fault METODOLOGIS (yang membatalkan kesahihan hasil) dan memperbaiki otomatis
HANYA yang objektif & reversibel. Fault data/hyperparameter hanya dilaporkan.

Auto-fix (butuh --apply, selalu backup .bak dulu, idempoten):
  - path backslash '\\' di manifest -> '/'.
  - RUN_NAME yang menimpa head_runs/<RUN> yang sudah ada -> sarankan suffix unik.

Deteksi lapor-saja (butuh tindakan manual):
  - leakage rel_path antar split, file dedup bocor ke manifest, label corrupt, path absolut,
    CV tanpa group orig_idx, INPUT_NORM mencurigakan, indikasi seleksi via test.

Exit 0 bila tidak ada fault manual tersisa; 1 bila ada.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

VALID_LABELS = {"Recylable", "Organik", "B3"}
SPLITS = ("train", "val", "test")


class Report:
    def __init__(self):
        self.fixed = []
        self.would_fix = []
        self.manual = []

    def fixed_(self, msg):
        self.fixed.append(msg)

    def would(self, msg):
        self.would_fix.append(msg)

    def man(self, msg):
        self.manual.append(msg)

    def has_unresolved(self):
        return bool(self.manual or self.would_fix)


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


# ---------------------- manifest audits ----------------------
def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def audit_manifests(manifest_dir: Path, apply: bool, rep: Report):
    rows_by_split = {}
    for split in SPLITS:
        p = manifest_dir / f"{split}.csv"
        if not p.exists():
            rep.man(f"Manifest hilang: {p}")
            continue
        rows = read_rows(p)
        if not rows:
            rep.man(f"Manifest kosong: {p}")
            continue
        rows_by_split[split] = (p, rows)

    # 1) backslash path -> auto-fix
    for split, (p, rows) in rows_by_split.items():
        header, data = rows[0], rows[1:]
        try:
            rp_idx = header.index("rel_path")
        except ValueError:
            rep.man(f"{p.name}: tak ada kolom 'rel_path'.")
            continue
        n_back = sum(1 for r in data if len(r) > rp_idx and "\\" in r[rp_idx])
        if n_back:
            if apply:
                backup(p)
                for r in data:
                    if len(r) > rp_idx:
                        r[rp_idx] = r[rp_idx].replace("\\", "/")
                with p.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([header] + data)
                rep.fixed_(f"{p.name}: {n_back} path backslash -> '/' (backup {p.name}.bak).")
                # refresh in-memory
                rows_by_split[split] = (p, [header] + data)
            else:
                rep.would(f"{p.name}: {n_back} path pakai backslash; jalankan --apply untuk ubah ke '/'.")

    # 2) path absolut + label corrupt -> lapor
    for split, (p, rows) in rows_by_split.items():
        header, data = rows[0], rows[1:]
        rp_idx = header.index("rel_path") if "rel_path" in header else 0
        lb_idx = header.index("label") if "label" in header else 1
        for i, r in enumerate(data, start=2):
            if len(r) <= max(rp_idx, lb_idx):
                continue
            rel = r[rp_idx].strip()
            lab = r[lb_idx].strip()
            if rel.startswith("/") or (len(rel) > 1 and rel[1] == ":"):
                rep.man(f"{p.name} baris {i}: path absolut '{rel}' (harus relatif ke Dataset/).")
            if lab and lab not in VALID_LABELS:
                rep.man(f"{p.name} baris {i}: label corrupt '{lab}'.")

    # 3) leakage antar split -> lapor
    seen = {}
    for split, (p, rows) in rows_by_split.items():
        header, data = rows[0], rows[1:]
        rp_idx = header.index("rel_path") if "rel_path" in header else 0
        for r in data:
            if len(r) <= rp_idx:
                continue
            rel = r[rp_idx].strip().replace("\\", "/")
            if not rel:
                continue
            if rel in seen and seen[rel] != split:
                rep.man(f"LEAKAGE: '{rel}' ada di split '{seen[rel]}' dan '{split}' "
                        "(perbaiki split di preprocessing_lokal.ipynb).")
            seen[rel] = split

    # 4) file dedup bocor ke manifest -> lapor
    dedup = manifest_dir / "dedup_log.csv"
    if dedup.exists():
        removed = set()
        with dedup.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fp = (row.get("file_dihapus") or "").strip().replace("\\", "/")
                if fp:
                    removed.add(fp)
        for rel in seen:
            if rel in removed:
                rep.man(f"'{rel}' tercatat dihapus di dedup_log.csv tapi masih di manifest.")


# ---------------------- notebook/config audits ----------------------
def _notebook_text(nb_path: Path) -> str:
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as e:
        return ""
    chunks = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def audit_notebook(nb_path: Path, manifest_dir: Path, rep: Report, root: Path, apply: bool):
    if not nb_path.exists():
        rep.man(f"Notebook tidak ditemukan: {nb_path}")
        return
    text = _notebook_text(nb_path)
    if not text:
        rep.man(f"Gagal membaca sel notebook: {nb_path}")
        return

    # RUN_NAME menimpa run yang sudah ada
    m = re.search(r"RUN_NAME\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m:
        run_name = m.group(1)
        run_dir = root / "head_runs" / run_name
        if run_dir.exists() and any(run_dir.iterdir()):
            suggestion = f"{run_name}_v2"
            rep.man(f"RUN_NAME='{run_name}' menunjuk head_runs/{run_name} yang SUDAH berisi hasil; "
                    f"ganti agar tidak tertimpa (mis. '{suggestion}'). (Edit manual di Cell config.)")
    else:
        rep.man("RUN_NAME tidak ditemukan di notebook; pastikan tiap eksperimen punya nama unik.")

    # CV tanpa group orig_idx
    uses_cv = re.search(r"\b(KFold|StratifiedKFold|cross_val_score|cross_validate)\b", text)
    uses_group = re.search(r"\b(GroupKFold|StratifiedGroupKFold|groups\s*=|group\s*=)\b", text) \
        or "orig_idx" in text and re.search(r"group", text)
    if uses_cv and not re.search(r"GroupKFold|StratifiedGroupKFold|groups\s*=", text):
        rep.man("Terdeteksi CV (KFold/cross_val) tanpa GroupKFold/groups=orig_idx; "
                "view augmentasi bisa bocor antar fold. Group WAJIB by orig_idx.")

    # INPUT_NORM mencurigakan
    mn = re.search(r"INPUT_NORM\s*=\s*['\"]([^'\"]+)['\"]", text)
    if mn and mn.group(1) != "standardize":
        rep.man(f"INPUT_NORM='{mn.group(1)}' (bukan 'standardize'); pastikan statistik di-fit dari "
                "TRAIN saja agar tidak bocor dari val/test.")

    # indikasi seleksi via test (heuristik): memilih berdasarkan test_metrics sebelum final
    if re.search(r"argmax[^\n]*test|sort[^\n]*test_(f1|recall|acc)", text):
        rep.man("Heuristik: ada pengurutan/argmax berbasis metrik TEST; seleksi model harus pakai VAL, "
                "test hanya sekali untuk model final.")


# ------------------------------ main ------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", default="preprocessing_output")
    ap.add_argument("--notebook", default="siglip_embedding_probe_colab.ipynb")
    ap.add_argument("--root", default=".")
    ap.add_argument("--apply", action="store_true",
                    help="terapkan auto-fix aman (tanpa ini: dry-run)")
    args = ap.parse_args()

    root = Path(args.root)
    rep = Report()

    man_dir = Path(args.manifest_dir)
    if man_dir.exists():
        audit_manifests(man_dir, args.apply, rep)
    else:
        rep.man(f"Folder manifest tidak ada: {man_dir}")

    audit_notebook(Path(args.notebook), man_dir, rep, root, args.apply)

    print("================ METHODOLOGY FAULT FIXER ================")
    print(f"mode: {'APPLY (menulis perbaikan aman)' if args.apply else 'DRY-RUN (tidak mengubah file)'}")

    if rep.fixed:
        print("\n[FIXED] (sudah diperbaiki)")
        for m in rep.fixed:
            print(f"  - {m}")
    if rep.would_fix:
        print("\n[WOULD-FIX] (jalankan --apply untuk menerapkan)")
        for m in rep.would_fix:
            print(f"  - {m}")
    if rep.manual:
        print("\n[MANUAL] (butuh keputusan/perbaikan manusia)")
        for m in rep.manual:
            print(f"  - {m}")

    print("\n--- RINGKASAN ---")
    print(f"  auto-fixed      : {len(rep.fixed)}")
    print(f"  menunggu --apply: {len(rep.would_fix)}")
    print(f"  manual          : {len(rep.manual)}")
    if not rep.has_unresolved() and not rep.fixed:
        print("  [OK] Tidak ada fault metodologis terdeteksi.")
    elif not rep.has_unresolved():
        print("  [OK] Semua fault yang terdeteksi sudah diperbaiki otomatis.")
    else:
        print("  [ACTION] Masih ada fault yang perlu ditangani.")

    return 1 if rep.has_unresolved() else 0


if __name__ == "__main__":
    sys.exit(main())
