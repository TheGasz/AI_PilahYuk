# Tech

## Stack
- Python, Jupyter notebooks (no package manifest, build system, or test suite).
- Local: pandas/CSV, image hashing (MD5 raw bytes, SHA256 on RGB pixels, dHash perceptual), PIL-style image handling.
- Colab/local GPU: Hugging Face `google/siglip-so400m-patch14-384` (frozen vision encoder, 1152-d `pooler_output`); for sub-labeling the matching **text tower** of the same checkpoint is used for zero-shot.
- Heads (PyTorch): `linear | mlp | mlp_norm | mlp2 | resmlp` (RMSNorm, GELU/SiLU, dropout). Classic ML via scikit-learn (`LogisticRegression(class_weight='balanced')`, kNN cosine, SVM RBF), LightGBM, k-means (sub-label clustering).
- Reports: `python-docx` scripts (generated via the `docx-report-generator` skill).

## Pipeline
1. `EDA/EDA_Senpro.ipynb`: exploratory analysis, outputs CSV/PNG in `EDA/`.
2. `preprocessing_lokal.ipynb` (run locally): dedup (MD5 + pixel SHA256 + dHash), optional data-cleaning stage (apply manual review CSVs), quality filter, RGB convert, class weights, 70/15/15 stratified split, augmentation config. Writes `preprocessing_output/`.
3. `siglip_embedding_probe_colab.ipynb` (run on Colab **or** local GPU via tunnel; one `ENV` switch): extract embeddings to `.npz`, train/compare classifier heads, run data-cleaning diagnostics, and run the sub-labeling pipeline (stages S0–S6).

## Running the notebooks
- Notebooks run interactively in Jupyter/VS Code (local) or Colab. Do not start long-running training from the shell.
- `siglip_embedding_probe_colab.ipynb` runs local or Colab via a single `ENV` variable (`'auto'|'local'|'colab'`). `auto` picks `ROOT_COLAB` (Drive) on Colab, else `ROOT_LOCAL` (repo path). pip install only runs on Colab.
- `.docx` reports are produced by committed `python-docx` scripts, not by hand-editing the files. Use the `docx-report-generator` skill to create/update them; read numbers from `preprocessing_output/*.csv` and `class_weights.json` rather than hardcoding.

## Conventions
- Augmentation is applied to Train only; Val/Test stay clean to avoid leakage.
- Manifests (`train.csv`, `val.csv`, `test.csv`, columns `rel_path,label`) use relative paths with forward slashes `/` for Linux/Drive portability.
- Duplicates are moved to `Dataset/duplicates_removed/`, never deleted. Cleaning decisions (`buang`/`relabel:<Kelas>`/`ok`/`setuju`) are applied by moving files, logged to `cleaning_log.csv`.
- Track per-class Recall/Precision/F1 each epoch; early stop on validation loss. Loss defaults to CrossEntropy + label smoothing (Focal optional); mixup in embedding space is available.
- Use class weights from `preprocessing_output/class_weights.json`; `CLASS_WEIGHT_MODE='effective'` computes `N/(C*N_c)` from train `.npz` row counts (accounts for per-class `N_AUG`).
- Bump `DATA_VERSION` for each new dataset version so embeddings land in a new `EMBEDDING_DIR` and `RUN_NAME`; never overwrite old results.
- Git: work happens on feature branches (e.g. `Data_Preprocessing`); `Dataset/` and `Archive/` are gitignored.
