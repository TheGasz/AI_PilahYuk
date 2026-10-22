# Structure

```
.
├── Dataset/                  # Clean working data (gitignored)
│   ├── 0_Recylable/
│   ├── 1_Organik/
│   ├── 2_B3/
│   └── duplicates_removed/   # Quarantined duplicates + cleaning "buang"
├── Archive/                  # Raw source datasets (gitignored, read-only reference)
├── EDA/                      # EDA notebook + CSV/PNG outputs (eda_*.csv, histograms)
├── preprocessing_output/     # train/val/test.csv (rel_path,label), class_weights.json,
│                             # augmentasi_config.json, dedup_log.csv, cleaning_log.csv,
│                             # metadata_kualitas.csv
├── embeddings/               # Extracted SigLIP .npz per embedding version (gitignored, large)
├── head_runs/                # Head checkpoints + history JSON per RUN_NAME
│   ├── mlp_head_v2_ce_ls_mixup/
│   ├── mlp_head_v3clean_ce_ls_mixup/
│   └── mlp_head_v3clean_ce_ls_mixup_lora_r8_b6/   # LoRA experiment (reverted, kept for record)
├── data_diagnostics/         # Per-embedding data-cleaning diagnostic CSVs (OOF, label noise,
│                             # near-dup cross-split, ambiguous, outliers)
├── sublabel_v2/              # Sub-labeling pipeline outputs (taxonomy v2): sublabels.csv,
│                             # taxonomy.json, zeroshot*, cluster_*, al_round_*, gold_status,
│                             # cek_kategori.csv, audit_log.csv, output_manifest.json, ...
├── sublabel_head_v2_v3clean_r1/   # Hierarchical sub-head checkpoints + comparison CSV
├── Output_nearestneighborEmbbed/  # kNN error analysis figures
├── preprocessing_lokal.ipynb      # Local cleaning + dedup (incl. dHash) + split
├── siglip_embedding_probe_colab.ipynb  # Embedding extract + head training + diagnostics + sub-labeling (S0-S6)
├── documentation.md          # Running project log (Indonesian)
└── *.docx                    # Generated reports (Laporan_*, Review_*)
```

Note: head/sub-label artifacts may also appear in root-level copies (`mlp_head_v2_ce_ls_mixup/`,
`mlp_head_v3clean_ce_ls_mixup/`) alongside `head_runs/`, e.g. when exported from Colab Drive.

## Rules
- Never modify or delete files in `Dataset/` or `Archive/` without asking; they are large and untracked.
- Generated artifacts live in `preprocessing_output/`, `EDA/`, `embeddings/`, `head_runs/`,
  `data_diagnostics/`, `sublabel_*`, and `Output_*` folders. Regenerate via notebooks instead of
  hand-editing. The sub-labeling pipeline never writes to `Dataset/` or `Archive/` (read-only there).
- Update `documentation.md` when pipeline steps change.
- Report generator scripts (`generate_*.py`) belong at repo root in snake_case; create/update them
  via the `docx-report-generator` skill. (None are committed yet.)
- Bump the version suffix (`DATA_VERSION`, `TAXONOMY_VERSION`, `RUN_NAME`) for new experiments so
  prior outputs stay intact.
