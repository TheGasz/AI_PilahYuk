# Tech

## Stack
- Python, Jupyter notebooks (no package manifest, build system, or test suite).
- Local: pandas/CSV, image hashing (MD5 raw bytes, SHA256 on RGB pixels), PIL-style image handling.
- Colab: Hugging Face `google/siglip-so400m-patch14-384` (frozen vision encoder, 1152-d embeddings), PyTorch MLP head (Linear → GELU → Dropout → Linear), scikit-learn (`LogisticRegression(class_weight='balanced')`, kNN error analysis).
- Reports: `python-docx` scripts generate `.docx` files.

## Pipeline
1. `EDA/EDA_Senpro.ipynb`: exploratory analysis, outputs CSV/PNG in `EDA/`.
2. `preprocessing_lokal.ipynb` (run locally): dedup, quality filter, RGB convert, class weights, 70/15/15 stratified split, augmentation config. Writes `preprocessing_output/`.
3. `siglip_embedding_probe_colab.ipynb` (run on Colab): extract embeddings to `.npz` on Google Drive, then train classifier head.

## Commands
```powershell
python generate_dataset_report.py        # builds Laporan_Review_Dataset_Senpro.docx
python generate_pure_dataset_review.py   # builds Review_Murni_Dataset.docx
```
Notebooks are run interactively in Jupyter/VS Code (local) or Colab. Do not start long-running training from the shell.

## Conventions
- Augmentation is applied to Train only; Val/Test stay clean to avoid leakage.
- Manifests (`train.csv`, `val.csv`, `test.csv`) use relative paths with forward slashes `/` for Linux/Drive portability.
- Duplicates are moved to `Dataset/duplicates_removed/`, never deleted.
- Track per-class Recall/Precision/F1 each epoch; early stop on validation loss.
- Use class weights from `preprocessing_output/class_weights.json` for loss.
- Git: work happens on feature branches (e.g. `Data_Preprocessing`); `Dataset/` and `Archive/` are gitignored.
