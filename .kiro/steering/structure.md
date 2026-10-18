# Structure

```
.
├── Dataset/                  # Clean working data (gitignored)
│   ├── 0_Recylable/
│   ├── 1_Organik/
│   ├── 2_B3/
│   └── duplicates_removed/   # Quarantined duplicates
├── Archive/                  # Raw source datasets (gitignored, read-only reference)
├── EDA/                      # EDA notebook + CSV/PNG outputs (eda_*.csv, histograms)
├── preprocessing_output/     # train/val/test.csv, class_weights.json,
│                             # augmentasi_config.json, dedup_log.csv, metadata_kualitas.csv
├── Output_nearestneighborEmbbed/  # kNN error analysis figures
├── preprocessing_lokal.ipynb      # Local cleaning + split
├── siglip_embedding_probe_colab.ipynb  # Colab embedding + head training
├── generate_dataset_report.py     # python-docx report generator
├── generate_pure_dataset_review.py
├── documentation.md          # Running project log (Indonesian)
└── *.docx                    # Generated reports
```

## Rules
- Never modify or delete files in `Dataset/` or `Archive/` without asking; they are large and untracked.
- Generated artifacts live in `preprocessing_output/`, `EDA/`, and `Output_*` folders. Regenerate via notebooks instead of hand-editing.
- Update `documentation.md` when pipeline steps change.
- Keep new scripts at repo root next to the existing generators; keep naming in snake_case.
