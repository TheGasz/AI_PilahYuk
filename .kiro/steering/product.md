# Product

Senpro (UGM Semester 5 course project): 3-class waste image classification using frozen SigLIP embeddings.

## Classes
- `0_Recylable` (spelling in folder/label name is intentional, do not "fix" it)
- `1_Organik`
- `2_B3` (hazardous waste; minority class, Recall B3 is the priority metric)

Label index: `{'Recylable': 0, 'Organik': 1, 'B3': 2}`

## Current Goal
1. Evaluate the model for good classification: pick heads by **val** metrics (macro F1, per-class Recall/Precision/F1, B3 Recall with B3 Precision as a false-alarm check), and touch **test** only once for the chosen model.
2. Use the SigLIP embedding space (from `siglip_embedding_probe_colab.ipynb`) to find and clean dataset problems: misclassified samples, label noise, near-duplicates, and visually ambiguous pairs. Dataset fixes feed back into `preprocessing_lokal.ipynb`, then embeddings are re-extracted into a new embedding dir.
3. Auto-label sub-classes (material/object type) on top of the 3 coarse classes using a hybrid zero-shot SigLIP + clustering + active learning pipeline (taxonomy v2, implemented inside the probe notebook). See the `solusi-auto-labeling-siglip` steering.

## Current Results

### v2 run (`mlp_head_v2_ce_ls_mixup`, data v2 = 515 dups removed)
- Final model chosen: **`ens_lintas_model`** (MLP multi-seed + kNN cosine + SVM RBF). Selected on **val**; test touched once.
- Val macro F1 ~0.9913 (tied with best `ens_seed_resmlp` 0.9915) but higher B3 precision; `macro_F1` vs `macro_F1_nodup` differ only ~0.0001-0.0002 (near-dup val does not inflate scores).
- Test: macro F1 ~0.9936, B3 recall 0.9942, B3 precision 0.9961 (516/520 B3 correct, 0 false alarms into B3). Residual error concentrated in **Recylable <-> Organik**.
- Effective class weights fixed a double-counting bug (B3 weight 2.90 -> 1.62) without hurting B3 recall.
- P2 bias calibration tried and **rejected**: improved val marginally but lowered test; it only moves R<->O error, not reduces it. Cell 21 disabled (`P2_ENABLED=False`).

### v3clean run (`mlp_head_v3clean_ce_ls_mixup`, data cleaned)
- Data cleaning applied: dHash dedup (1174 dups moved vs 515), 49 relabels + 17 drops from R<->O review. New split: Train 20275, Val/Test 4345 each (B3 477 per eval split). Test per-class: Recylable 1843, Organik 2025, B3 477. Class weights: Recylable 0.7859, Organik 0.7152, B3 3.0362.
- `ens_lintas_model` val macro F1 ~0.9956.
- LoRA fine-tune of the backbone was tried and **reverted**: val macro F1 (0.9883) was *worse* than the frozen-embedding probe and B3 recall dropped. Frozen backbone + head probing wins; residual R<->O error is intrinsic label ambiguity, not a backbone capacity limit.

Heads are all near 99.5%, so remaining gains come from data cleaning, not a bigger head.

## Notebook Details

### EDA (`EDA/EDA_Senpro.ipynb`)
- Run on raw `Dataset/`; set `ROOT_DIR` in the config cell (auto-detects Colab vs local).
- Sections: class distribution + imbalance ratio; pixel-exact duplicates (SHA256 of size+RGB bytes); technical properties (resolution, aspect ratio, format, color mode); quality (corrupt via `verify()`, blur by Laplacian variance < 100, dark mean < 40, bright mean > 220, MD5 duplicates); HSV histograms per class with intersection and Bhattacharyya (overlap_tinggi if < 0.3).
- Findings: ~1000 identical duplicates, mixed color modes, widely varying resolution, B3 minority.
- Outputs are `eda_*.csv` / `eda_histogram_*.png` in `EDA/`.

### Preprocessing (`preprocessing_lokal.ipynb`, local only)
- Config: `MIN_RESOLUTION=50`, `DARK_THRESHOLD=50`, `BRIGHT_THRESHOLD=220`, split 70/15/15, `RANDOM_SEED=42`, `DHASH_MAX_HAMMING=5`.
- Stage 1 dedup: MD5 (raw bytes) + pixel SHA256 + **dHash** (8x8 difference hash, grouped within the same class, merged by Hamming <= threshold to catch cross-format dups that slip past exact hashes); keep highest-resolution representative; move the rest to `Dataset/duplicates_removed/` and log to `dedup_log.csv`.
- Stage 1.5 data cleaning (between dedup and quality): reads manual-review CSVs (`CLEANING_CSVS`; columns `path` + `keputusan`: `buang` / `relabel:<Kelas>` / `setuju` (use `saran_relabel`) / `ok`) and applies them permanently by moving files (`buang` -> `duplicates_removed/`, relabel -> target class folder). Idempotent, logs to `cleaning_log.csv`. `CROSS_SPLIT_DUP_CSV` moves cross-split near-dups into Train at split time so Val/Test stay shadow-free.
- Stage 2 quality: drop corrupt and < 50x50; convert non-RGB to RGB **in place (overwrites file)**; dark/overexposed are flagged only, not dropped (`metadata_kualitas.csv`).
- Stage 3: class weights `N_total / (N_classes * N_c)` -> `class_weights.json`; augmentation recipe -> `augmentasi_config.json` (general: flips, ColorJitter, RandomResizedCrop 384 scale 0.8-1.0; B3 extra: rotation 30, stronger jitter, scale 0.7-1.0, grayscale 0.05).
- Stage 4: stratified split (two-step `train_test_split`), asserts no overlap, manifests (`rel_path,label`) use `/` paths relative to `Dataset/`.

### Embedding probe (`siglip_embedding_probe_colab.ipynb`, Colab or local GPU)
- Runs local or Colab via `ENV` (`'auto'|'local'|'colab'`); Drive mount and pip install happen only on Colab.
- `DATA_VERSION` (e.g. `'v3clean'`) is appended to `EMBEDDING_DIR` and `RUN_NAME` so new dataset versions never overwrite old embeddings/results. `CELL 15` single-head test eval is guarded by `RUN_SINGLE_HEAD_TEST=False` so Run All touches test only once (Cell 19, final model).
- Backbone: `google/siglip-so400m-patch14-384`, vision tower only, frozen, fp16, `pooler_output` 1152-d.
- Stage 1 (run once): each train image becomes `1 + N_AUG` views, `N_AUG = {Recylable: 2, Organik: 2, B3: 5}`. Val/test use view 0 only. Saves `embeddings_{train,val,test}.npz` under `embeddings/<model>_naug_2_2_5[_<DATA_VERSION>]/` with keys `embeddings, labels, label_names, paths, orig_idx, view_idx`. Chunked and resumable.
- Stage 2 (seconds per run, no SigLIP call): input norm `standardize` (stats from train only); heads `linear | mlp | mlp_norm | mlp2 | resmlp`; default loss CrossEntropy + label smoothing 0.05 (Focal optional, gamma 2) with effective class weights; optional embedding-space mixup (`MIXUP_ALPHA`); early stopping patience 8 on val loss; history JSON and checkpoints saved to `head_runs/<RUN_NAME>/`.
- Comparison cells: neural heads, kNN cosine (k=15, temp 0.05, vote x class weight), SVM RBF (C from grid via val) and LightGBM (CPU, slow), ensembles (all heads, multi-seed of best, `ens_lintas_model` = MLP multi-seed + kNN + SVM). Table ranked by val macro F1 (plus `macro_F1_nodup`, OOF macro F1), saved as `perbandingan_head_<RUN_NAME>.csv`. `mark_test_use()` warns if test is viewed for more than one model.
- Data-cleaning diagnostics (Cell 12b/12c): OOF `StratifiedGroupKFold` predictions (grouped by `orig_idx`), label-noise candidates (confident learning + kNN), cross-split near-duplicates, ambiguous samples, centroid outliers; CSVs to `data_diagnostics/<embedding>/`. Error analysis runs on **val** (not test). Fixes feed back into `preprocessing_lokal.ipynb`.
- P2 bias calibration (Cell 21) exists but is disabled (`P2_ENABLED=False`).

### Sub-labeling (same notebook, stages S0-S6)
- Goal: assign material/object sub-labels within each coarse class using the frozen embeddings already on disk (no re-extraction). Zero-shot uses the **text tower** of the same so400m checkpoint; see the `solusi-auto-labeling-siglip` steering for the full method.
- Taxonomy is **v2** (implemented). Sub-label names are Indonesian (e.g. `Plastik_Kemasan`, `Baterai`, `Sisa_Makanan`); `lainnya` is the per-category open-set fallback. Each sub-label carries verbatim `tindakan_konsumen_pra_buang` and `rute_penyaluran_pengolahan` text that must come from the user (validation fails on the `<<ISI DARI USER>>` placeholder).
- Stages: S0 config + `taxonomy.json`; S1 filename-heuristic seed labels + gold set + spot check; S2 k-means clustering per category; S3 zero-shot constrained + global category check, calibrate tau on gold; S4 active learning with human-verified eval set; S5 write `sublabels.csv` + small-class policy (`latih`/`merge:<target>`/`rule_only`); S6 hierarchical sub-head / flat-head, select on val, test once.
- Outputs under `sublabel_<TAXONOMY_VERSION>/` (currently `sublabel_v2/`): `sublabels.csv`, `taxonomy.json`, `zeroshot*`, `cluster_*`, `al_round_*`, `gold_status.csv`, `cek_kategori.csv`, `audit_log.csv`, `output_manifest.json`. Sub-head checkpoints in `sublabel_head_<TAXONOMY_VERSION>_<DATA_VERSION>_r1/`.

## Rules for Changes
- Group CV or any resampling must group by `orig_idx` so augmented views of one image never leak across folds.
- Never select models using test metrics; touch test once for the chosen model.
- Change `RUN_NAME` for each new head experiment, `DATA_VERSION` for each dataset version, and `TAXONOMY_VERSION` for each sub-label taxonomy change, so old results are not overwritten.
- When the dataset changes, re-run preprocessing, re-extract embeddings into a new embedding dir, and update `documentation.md`.
- Sub-labeling: label is decided at view 0 per `orig_idx` and propagated to augmented views; pseudo-labels are never used as ground-truth evaluation (val/test sub-heads use only human-verified labels); never write to `Dataset/` or `Archive/`.
- Verbatim consumer-action / recycle-route text for sub-labels must come from the user, never invented.
- Project docs and notebook prose are in Indonesian; match that in existing files.
