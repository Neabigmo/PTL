# Phase 03 - Preprocessing and Harmonization Recovery

Last updated: 2026-04-25 14:45 Asia/Shanghai

## Scope

- Re-opened Phase 03 after the initial pilot was judged scientifically incomplete.
- Replaced the old "single processed h5ad" mindset with a staged preprocessing contract:
  - cell metadata
  - QC manifest
  - feature metadata
  - matrix-store manifest
  - pseudobulk expression table
  - delta signature table
  - optional convenience AnnData only when the write is operationally reasonable
- Reprocessed:
  - `NormanWeissman2019_filtered`
  - `ReplogleWeissman2022_K562_essential`
- Re-ran external validation discovery and adopted `GSE284197_screen` as the active external validation dataset.
- Reclassified `E-MTAB-14567` as an archived candidate rather than the active external holdout.

## External validation reset

### Archived candidate: E-MTAB-14567

- Source: BioStudies ArrayExpress
- Local directory: `H:\2026try\4.24\data\external\E-MTAB-14567`
- Download log: `results/logs/downloads/phase03_external_E-MTAB-14567_download.log`
- Why it was not adopted:
  - `filtered_feature_bc_matrix_rerun.h5` contains gene-expression features only.
  - `filtered_feature_bc_matrix.h5` retains guide-capture features, but the bundle does not expose a clean, reproducible per-cell perturbation mapping contract.
  - The available `idf` and `sdrf` metadata are not enough to promote the dataset into the main external-validation lane without additional reconstruction work.
- Final status in this phase: archived candidate, not active external validation.

### Adopted external validation dataset: GSE284197_screen

- Source: GEO supplementary files
- Local directory: `H:\2026try\4.24\data\external\GSE284197`
- Download log: `results/logs/downloads/phase03_external_GSE284197_download.log`
- Files acquired:
  - `GSE284197_screen.h5ad`
  - `GSE284197_feature_ref.csv.gz`
  - `GSE284197_feature_ref-illumina.csv.gz`
- The first download was interrupted at about 14%; it was completed successfully via resume (`curl -C - --retry-all-errors`).
- Final local size of `GSE284197_screen.h5ad`: `4,450,088,782` bytes.

## Why GSE284197_screen qualified

- The backed `.h5ad` is readable with shape `137,317 x 38,593`.
- `obs` directly exposes the fields needed for a scientific external holdout:
  - `perturbation`
  - `sgRNA`
  - `sgRNA_effective`
  - `Gene_target`
  - `Gene_target_single`
  - `batch`
  - `timepoint`
  - `species`
  - `stage`
- `species` is uniformly `human`.
- Controls are explicitly recoverable from `perturbation`:
  - `WT`
  - `NT`
- Single-target perturbations are recoverable from `Gene_target_single`.
- The matrix is backed sparse CSR, so it can be processed in chunks without dense in-memory rewriting.

## New preprocessing contract

- The reusable CLI remains `src/data/preprocess_dataset.py`, but the output contract is now staged and low-memory.
- New per-dataset primary artifacts:
  - `*_cell_metadata.parquet`
  - `*_qc_manifest.parquet`
  - `*_feature_metadata.parquet`
  - `*_matrix_store_manifest.json`
  - `*_pseudobulk.parquet`
  - `*_delta_signatures.parquet`
- Optional artifact:
  - `*_convenience_view.h5ad`
- Matrix handling decision:
  - The canonical matrix store is now `backed_source_with_qc_manifest`.
  - This preserves a lossless reconstruction path without forcing full dense matrix rewrites.
  - The contract is therefore unified across sparse and dense-backed inputs.

## Labeling and reference rules

- `NormanWeissman2019_filtered`
  - perturbation label from `perturbation`
  - control label: `control`
  - reference field: `batch`
- `ReplogleWeissman2022_K562_essential`
  - perturbation label from `perturbation`
  - control label: `control`
  - reference field: `batch`
- `GSE284197_screen`
  - control priority: `NT`, then `WT`
  - perturbation label logic:
    - use `perturbation` for controls
    - use `Gene_target_single` for single-target perturbed cells
    - collapse unresolved multi-target cells to `MULTI_TARGET`
  - contract filter:
    - keep only `human` cells
    - keep controls (`WT`, `NT`) and cells with non-`NA` `Gene_target_single`
  - reference fields:
    - `batch`
    - `timepoint`

## Reprocessing outputs

### NormanWeissman2019_filtered

- Input cells: `111,445`
- Contract-eligible cells: `111,445`
- Kept for analysis: `110,772`
- Unique perturbation labels kept: `237`
- Signature rows written: `1,829`
- Key files:
  - `data/processed/NormanWeissman2019_filtered_cell_metadata.parquet`
  - `data/processed/NormanWeissman2019_filtered_qc_manifest.parquet`
  - `data/processed/NormanWeissman2019_filtered_feature_metadata.parquet`
  - `data/processed/NormanWeissman2019_filtered_matrix_store_manifest.json`
  - `data/processed/NormanWeissman2019_filtered_pseudobulk.parquet`
  - `data/processed/NormanWeissman2019_filtered_delta_signatures.parquet`
  - `data/processed/NormanWeissman2019_filtered_convenience_view.h5ad`

### ReplogleWeissman2022_K562_essential

- Input cells: `310,385`
- Contract-eligible cells: `310,385`
- Kept for analysis: `51,821`
- Unique perturbation labels kept: `463`
- Signature rows written: `3,258`
- Key files:
  - `data/processed/ReplogleWeissman2022_K562_essential_cell_metadata.parquet`
  - `data/processed/ReplogleWeissman2022_K562_essential_qc_manifest.parquet`
  - `data/processed/ReplogleWeissman2022_K562_essential_feature_metadata.parquet`
  - `data/processed/ReplogleWeissman2022_K562_essential_matrix_store_manifest.json`
  - `data/processed/ReplogleWeissman2022_K562_essential_pseudobulk.parquet`
  - `data/processed/ReplogleWeissman2022_K562_essential_delta_signatures.parquet`
- Convenience AnnData was intentionally not re-written for the recovery contract because the backed-source manifest now serves as the canonical matrix representation.

### GSE284197_screen

- Input cells: `137,317`
- Contract-eligible cells: `52,020`
- Kept for analysis: `51,728`
- Unique perturbation labels kept: `46`
- Signature rows written: `73`
- Key files:
  - `data/processed/GSE284197_screen_cell_metadata.parquet`
  - `data/processed/GSE284197_screen_qc_manifest.parquet`
  - `data/processed/GSE284197_screen_feature_metadata.parquet`
  - `data/processed/GSE284197_screen_matrix_store_manifest.json`
  - `data/processed/GSE284197_screen_pseudobulk.parquet`
  - `data/processed/GSE284197_screen_delta_signatures.parquet`

## Notes on memory handling

- The recovery path avoids dense full-matrix rewrites.
- The canonical retained-cell representation is now:
  - source matrix on disk
  - row-level QC manifest
  - feature metadata
- Chunked normalization and accumulation are done during pseudobulk/signature generation only.
- This removes the earlier need to choose between a huge dense rewrite and a weaker summary-only fallback.

## Acceptance check

- `E-MTAB-14567` is no longer the primary external validation dataset.
- `GSE284197_screen` is downloaded, inspected, and processed under the same staged contract as the internal pilot datasets.
- `Norman` and `Replogle` now both have:
  - harmonized cell metadata
  - QC manifests
  - feature metadata
  - matrix-store manifests
  - pseudobulk tables
  - delta signature tables
- `results/tables/preprocessing_summary.csv` has been rewritten to the new staged-contract schema.
- No Phase 04 split construction, model training, calibration analysis, evaluation, or manuscript drafting was started.
