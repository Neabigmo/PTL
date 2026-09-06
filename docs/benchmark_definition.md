# Benchmark Definition

Last updated: 2026-04-25 Asia/Shanghai

## Purpose

Phase 04 defines the first reproducible context-stress benchmark split system for the Perturbation Transportability Lens project.

- The `signature` track is the canonical modeling interface for Phase 05 baselines.
- The `cell` track is the audit and support layer used for leakage checks, support accounting, and future cell-level extensions.

All split artifacts are generated from the Phase 03 recovery contract only:

- `*_cell_metadata.parquet`
- `*_qc_manifest.parquet`
- `*_delta_signatures.parquet`
- `results/tables/preprocessing_summary.csv`

## Canonical units

- Signature unit: one row in `*_delta_signatures.parquet`
- Cell unit: one retained cell row in `*_cell_metadata.parquet`, referenced back to the backed source matrix through `dataset_id + row_index` and the QC manifest

## Split families

### random_split

Random train / validation / test partition inside a dataset. This is the low-stress reference split.

### unseen_perturbation_split

The test split holds out complete perturbation labels from training. Validation comes from seen perturbations so that the stress remains concentrated on the test set.

### unseen_combination_split

The test split is restricted to true combinatorial perturbations detected from `perturbation_label` delimiters. Placeholder labels such as `MULTI_TARGET` are excluded.

### unseen_cell_state_split

The test split holds out cell-state labels when meaningful state variation exists.

- On the cell track, the preferred state field order is `cell_context`, then `predictions`, then `supervised_name`.
- On the signature track, state labels are derived from dominant per-group cell annotations with an 0.80 purity threshold. If the dominant-label criterion is not satisfied, the split is marked unsupported rather than silently skipped.

### unseen_cell_type_or_lineage_split

The test split holds out lineage labels when available.

- On the cell track, the preferred lineage field order is `class`, then `supervised_name`, then `cell_context`.
- On the signature track, lineage labels are derived from dominant per-group cell annotations with an 0.80 purity threshold.

### dataset_heldout_split

One dataset is held out as test while the remaining datasets are split into train and validation. This is the main dataset-level transport stress family.

### low_support_split

The test split is defined by low-support perturbation groups identified from signature-level backing cell counts (`n_cells`). The same low-support group keys are then projected onto the cell track.

### external_holdout

`GSE284197_screen` is treated as a dedicated external holdout that never mixes with internal random sampling. This role is separate from, and in addition to, its participation in `dataset_heldout_split`.

## Dataset support notes

- `AdamsonWeissman2016_GSM2406677_10X005`: split role `train`; cell-state column `unsupported`; cell-lineage column `unsupported`; signature-state source `unsupported`; signature-lineage source `unsupported`.
- `DatlingerBock2017`: split role `train`; cell-state column `unsupported`; cell-lineage column `unsupported`; signature-state source `unsupported`; signature-lineage source `unsupported`.
- `DatlingerBock2021`: split role `train`; cell-state column `unsupported`; cell-lineage column `unsupported`; signature-state source `unsupported`; signature-lineage source `unsupported`.
- `DixitRegev2016_K562_TFs_7_days`: split role `train`; cell-state column `unsupported`; cell-lineage column `unsupported`; signature-state source `unsupported`; signature-lineage source `unsupported`.
- `GSE284197_screen`: split role `external_validation`; cell-state column `cell_context`; cell-lineage column `class`; signature-state source `unsupported`; signature-lineage source `unsupported`.
- `NormanWeissman2019_filtered`: split role `train`; cell-state column `unsupported`; cell-lineage column `unsupported`; signature-state source `unsupported`; signature-lineage source `unsupported`.
- `PapalexiSatija2021_eccite_RNA`: split role `train`; cell-state column `unsupported`; cell-lineage column `unsupported`; signature-state source `unsupported`; signature-lineage source `unsupported`.
- `ReplogleWeissman2022_K562_essential`: split role `train`; cell-state column `unsupported`; cell-lineage column `unsupported`; signature-state source `unsupported`; signature-lineage source `unsupported`.
- `ReplogleWeissman2022_rpe1`: split role `train`; cell-state column `unsupported`; cell-lineage column `unsupported`; signature-state source `unsupported`; signature-lineage source `unsupported`.

## Why `GSE284197_screen` has two roles

`GSE284197_screen` is both:

1. a standalone external holdout (`external_holdout`)
2. one of the held-out domains in `dataset_heldout_split`

These two roles answer different questions:

- `external_holdout` asks whether a model trained only on the internal scPerturb pool transfers to an independently sourced dataset.
- `dataset_heldout_split` asks whether dataset identity itself behaves like a transport boundary when all currently adopted datasets are treated symmetrically.

## Supported artifacts observed in this run

- `dataset_heldout_split` / `cell` / `all_datasets`
- `dataset_heldout_split` / `signature` / `all_datasets`
- `external_holdout` / `cell` / `internal_vs_external`
- `external_holdout` / `signature` / `internal_vs_external`
- `low_support_split` / `cell` / `AdamsonWeissman2016_GSM2406677_10X005`
- `low_support_split` / `cell` / `DatlingerBock2017`
- `low_support_split` / `cell` / `DatlingerBock2021`
- `low_support_split` / `cell` / `DixitRegev2016_K562_TFs_7_days`
- `low_support_split` / `cell` / `GSE284197_screen`
- `low_support_split` / `cell` / `NormanWeissman2019_filtered`
- `low_support_split` / `cell` / `PapalexiSatija2021_eccite_RNA`
- `low_support_split` / `cell` / `ReplogleWeissman2022_K562_essential`
- `low_support_split` / `cell` / `ReplogleWeissman2022_rpe1`
- `low_support_split` / `signature` / `DatlingerBock2017`
- `low_support_split` / `signature` / `DatlingerBock2021`
- `low_support_split` / `signature` / `DixitRegev2016_K562_TFs_7_days`
- `low_support_split` / `signature` / `GSE284197_screen`
- `low_support_split` / `signature` / `NormanWeissman2019_filtered`
- `low_support_split` / `signature` / `PapalexiSatija2021_eccite_RNA`
- `low_support_split` / `signature` / `ReplogleWeissman2022_K562_essential`
- `low_support_split` / `signature` / `ReplogleWeissman2022_rpe1`
- `random_split` / `cell` / `AdamsonWeissman2016_GSM2406677_10X005`
- `random_split` / `cell` / `DatlingerBock2017`
- `random_split` / `cell` / `DatlingerBock2021`
- `random_split` / `cell` / `DixitRegev2016_K562_TFs_7_days`
- `random_split` / `cell` / `GSE284197_screen`
- `random_split` / `cell` / `NormanWeissman2019_filtered`
- `random_split` / `cell` / `PapalexiSatija2021_eccite_RNA`
- `random_split` / `cell` / `ReplogleWeissman2022_K562_essential`
- `random_split` / `cell` / `ReplogleWeissman2022_rpe1`
- `random_split` / `signature` / `DatlingerBock2017`
- `random_split` / `signature` / `DatlingerBock2021`
- `random_split` / `signature` / `DixitRegev2016_K562_TFs_7_days`
- `random_split` / `signature` / `GSE284197_screen`
- `random_split` / `signature` / `NormanWeissman2019_filtered`
- `random_split` / `signature` / `PapalexiSatija2021_eccite_RNA`
- `random_split` / `signature` / `ReplogleWeissman2022_K562_essential`
- `random_split` / `signature` / `ReplogleWeissman2022_rpe1`
- `unseen_cell_state_split` / `cell` / `GSE284197_screen`
- `unseen_cell_type_or_lineage_split` / `cell` / `GSE284197_screen`
- `unseen_combination_split` / `cell` / `AdamsonWeissman2016_GSM2406677_10X005`
- `unseen_combination_split` / `cell` / `NormanWeissman2019_filtered`
- `unseen_combination_split` / `signature` / `NormanWeissman2019_filtered`
- `unseen_perturbation_split` / `cell` / `AdamsonWeissman2016_GSM2406677_10X005`
- `unseen_perturbation_split` / `cell` / `DatlingerBock2017`
- `unseen_perturbation_split` / `cell` / `DatlingerBock2021`
- `unseen_perturbation_split` / `cell` / `DixitRegev2016_K562_TFs_7_days`
- `unseen_perturbation_split` / `cell` / `GSE284197_screen`
- `unseen_perturbation_split` / `cell` / `NormanWeissman2019_filtered`
- `unseen_perturbation_split` / `cell` / `PapalexiSatija2021_eccite_RNA`
- `unseen_perturbation_split` / `cell` / `ReplogleWeissman2022_K562_essential`
- `unseen_perturbation_split` / `cell` / `ReplogleWeissman2022_rpe1`
- `unseen_perturbation_split` / `signature` / `DatlingerBock2017`
- `unseen_perturbation_split` / `signature` / `DatlingerBock2021`
- `unseen_perturbation_split` / `signature` / `DixitRegev2016_K562_TFs_7_days`
- `unseen_perturbation_split` / `signature` / `GSE284197_screen`
- `unseen_perturbation_split` / `signature` / `NormanWeissman2019_filtered`
- `unseen_perturbation_split` / `signature` / `PapalexiSatija2021_eccite_RNA`
- `unseen_perturbation_split` / `signature` / `ReplogleWeissman2022_K562_essential`
- `unseen_perturbation_split` / `signature` / `ReplogleWeissman2022_rpe1`

## Unsupported logic

Any infeasible split family is written explicitly as an `unsupported` JSON artifact and an `unsupported` row in `results/tables/split_audit.csv`. This prevents silent simplification and keeps the benchmark surface auditable.
