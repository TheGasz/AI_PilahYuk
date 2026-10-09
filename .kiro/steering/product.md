# Product

Senpro (UGM Semester 5 course project): 3-class waste image classification using frozen SigLIP embeddings.

## Classes
- `0_Recylable` (spelling in folder/label name is intentional, do not "fix" it)
- `1_Organik`
- `2_B3` (hazardous waste; minority class, Recall B3 is the priority metric)

Label index: `{'Recylable': 0, 'Organik': 1, 'B3': 2}`

## Current Goal
1. Evaluate the model for good classification: pick heads by **val** metrics (macro F1, per-class Recall/Precision/F1, B3 Recall with B3 Precision as a false-alarm check), and touch **test** only once for the chosen model.
2. Use the SigLIP embedding space (from `siglip_embedding_probe_colab.ipynb`) to find and clean dataset problems: misclassified samples, label noise, near-duplicates, and visually ambiguous pairs. Dataset fixes feed back into `preprocessing_lokal.ipynb`, then embeddings are re-extracted.

## Current Results (test set, 4524 images: Recylable 1942, Organik 2062, B3 520)
- Baseline MLP head: acc 99.14%, B3 recall 0.9846.
- Val ranking by macro F1: `svm_rbf` 0.9894 > `ens_seed_mlp2` 0.9890 > `knn_cosine` 0.9880 > `mlp2` 0.9879 > `linear` 0.9757 (worst).
- SVM on test: main residual error is Recylable <-> Organik (12 Recylable predicted Organik, 15 Organik predicted Recylable); B3 almost never confused (7 B3 -> Recylable).
- Heads are all near 99%, so remaining gains likely come from data cleaning, not a bigger head.

## Notebook Details

### EDA (`EDA/EDA_Senpro.ipynb`)
- Run on raw `Dataset/`; set `ROOT_DIR` in the config cell (auto-detects Colab vs local).
- Sections: class distribution + imbalance ratio; pixel-exact duplicates (SHA256 of size+RGB bytes); technical properties (resolution, aspect ratio, format, color mode); quality (corrupt via `verify()`, blur by Laplacian variance < 100, dark mean < 40, bright mean > 220, MD5 duplicates); HSV histograms per class with intersection and Bhattacharyya (overlap_tinggi if < 0.3).
- Findings: ~1000 identical duplicates, mixed color modes, widely varying resolution, B3 minority.
- Outputs are `eda_*.csv` / `eda_histogram_*.png` in `EDA/`.

### Preprocessing (`preprocessing_lokal.ipynb`, local only)
- Config: `MIN_RESOLUTION=50`, `DARK_THRESHOLD=50`, `BRIGHT_THRESHOLD=220`, split 70/15/15, `RANDOM_SEED=42`.
- Stage 1 dedup: MD5 (raw bytes) + pixel SHA256; keep highest-resolution representative; move the rest to `Dataset/duplicates_removed/` (515 moved) and log to `dedup_log.csv`.
- Stage 2 quality: drop corrupt and < 50x50; convert non-RGB to RGB **in place (overwrites file)**; dark/overexposed are flagged only, not dropped (`metadata_kualitas.csv`).
- Stage 3: class weights `N_total / (N_classes * N_c)` -> `class_weights.json`; augmentation recipe -> `augmentasi_config.json` (general: flips, ColorJitter, RandomResizedCrop 384 scale 0.8-1.0; B3 extra: rotation 30, stronger jitter, scale 0.7-1.0, grayscale 0.05).
- Stage 4: stratified split (two-step `train_test_split`), asserts no overlap, manifests use `/` paths relative to `Dataset/`.

### Embedding probe (`siglip_embedding_probe_colab.ipynb`, Colab GPU)
- Backbone: `google/siglip-so400m-patch14-384`, vision tower only, frozen, fp16, `pooler_output` 1152-d.
- Stage 1 (run once): each train image becomes `1 + N_AUG` views, `N_AUG = {Recylable: 2, Organik: 2, B3: 5}` (21107 train images -> 70599 rows). Val/test use view 0 only. Saves `embeddings_{train,val,test}.npz` under `ROOT/embeddings/<model>_naug_2_2_5/` with keys `embeddings, labels, label_names, paths, orig_idx, view_idx`. Chunked and resumable after disconnect.
- Stage 2 (seconds per run, no SigLIP call): input norm `standardize` (stats from train only); heads `linear | mlp | mlp_norm | mlp2 | resmlp` (RMSNorm, GELU/SiLU, dropout 0.3); Focal Loss (gamma 2) with class weights; early stopping patience 8 on val loss; history JSON and checkpoints saved to `ROOT/head_runs/<RUN_NAME>/`.
- Comparison cells: neural heads, kNN cosine (k=15, temp 0.05, vote x class weight), SVM RBF and LightGBM (CPU, slow), ensembles (all heads, multi-seed of best head). Table ranked by val macro F1, saved as `perbandingan_head_<RUN_NAME>.csv`; `report_test(name)` runs the single test evaluation.
- Data-cleaning cells: error analysis of Recylable predicted as Organik, and nearest-neighbor view of the closest Organik train images (view 0 only, cosine) for each miss; saved as `error_analysis_knn.png`.

## Rules for Changes
- Group CV or any resampling must group by `orig_idx` so augmented views of one image never leak across folds.
- Never select models using test metrics.
- Change `RUN_NAME` for each new experiment so old results are not overwritten.
- When the dataset changes, re-run preprocessing, re-extract embeddings into a new embedding dir, and update `documentation.md`.
- Project docs and notebook prose are in Indonesian; match that in existing files.
