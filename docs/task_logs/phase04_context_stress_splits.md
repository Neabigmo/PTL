# Phase 04 - Context-Stress Split System

Last updated: 2026-04-25 15:59 Asia/Shanghai

## Scope

- Implement the first formal Phase 04 benchmark split system on top of the Phase 03 recovery artifacts.
- Keep the benchmark dual-track:
  - `signature` track for canonical Phase 05 modeling
  - `cell` track for leakage checks, support accounting, and future cell-level extensions
- Preserve `GSE284197_screen` in two roles:
  - dedicated `external_holdout`
  - symmetric member of `dataset_heldout_split`

## Skill gate and skills used

- Checked the Phase 04 task against `SKILLS_INDEX.md`.
- User explicitly added and enabled:
  - `scikit-learn`
  - `anndata`
- Applied these skills as follows:
  - `scikit-learn`: deterministic `train_test_split`-based split construction and group-holdout logic
  - `anndata`: kept the cell track tied to the backed-source matrix contract through `row_index`, `obs_name`, and the QC manifest rather than inventing a second filtered matrix store

## Inputs used

- `PROJECT_RULES.md`
- `docs/task_packets/04_context_stress_splits.md`
- `docs/project_status.md`
- `docs/file_registry.md`
- `results/tables/preprocessing_summary.csv`
- Phase 03 recovery artifacts only:
  - `*_cell_metadata.parquet`
  - `*_qc_manifest.parquet`
  - `*_feature_metadata.parquet`
  - `*_delta_signatures.parquet`

## Main implementation

### New CLI

- Added `src/splits/make_splits.py`.
- The CLI:
  - loads the Phase 03 recovery summary
  - constructs split candidates for each dataset and both tracks
  - writes JSON split artifacts into `data/processed/splits/`
  - writes `results/tables/split_audit.csv`
  - rewrites `docs/benchmark_definition.md`

### Canonical split units

- Signature unit:
  - one row in `*_delta_signatures.parquet`
- Cell unit:
  - one retained row in `*_cell_metadata.parquet`
  - referenced back to source matrices through `dataset_id + row_index`
  - `obs_name` remains resolvable through the cell metadata parquet and QC manifest

### Split families implemented

Ready families:

- `random_split`
- `unseen_perturbation_split`
- `unseen_combination_split` for Norman only
- `low_support_split`
- `dataset_heldout_split`
- `external_holdout`
- `unseen_cell_state_split` on the `GSE284197_screen` cell track
- `unseen_cell_type_or_lineage_split` on the `GSE284197_screen` cell track

Explicit unsupported families were still written as JSON and audit rows, rather than being skipped silently.

## Dataset-specific behavior

### NormanWeissman2019_filtered

- Supports:
  - random
  - unseen perturbation
  - unseen combination
  - low support
- Does not support:
  - cell-state holdout
  - lineage holdout
- Reason:
  - `cell_context` is constant `K562`

### ReplogleWeissman2022_K562_essential

- Supports:
  - random
  - unseen perturbation
  - low support
- Does not support:
  - unseen combination
  - cell-state holdout
  - lineage holdout
- Reasons:
  - no true combinatorial perturbation regime in the retained labels
  - `cell_context` is constant `K562`

### GSE284197_screen

- Supports:
  - random
  - unseen perturbation
  - low support
  - cell-track unseen state
  - cell-track unseen lineage
  - external holdout
  - dataset-heldout
- Does not support:
  - signature-track unseen state
  - signature-track unseen lineage
  - unseen combination
- Reasons:
  - dominant-label purity at the signature-group level collapses to a single effective label, so a signature-track state/lineage holdout would not be scientifically meaningful
  - no true combinatorial perturbation regime after the Phase 03 contract filter

## Important semantic decisions written into the code

- Perturbation stress:
  - controls stay in the seen pool
- State / lineage stress on the cell track:
  - controls from the held-out state or lineage move with the held-out context into test
  - this prevents context leakage
- Low-support stress:
  - derived from the bottom 20% of non-control signature groups ranked by backing cell count `n_cells`
  - the same low-support `group_key` set is then projected onto the cell track
- External holdout:
  - `GSE284197_screen` never mixes with internal random sampling in `external_holdout`

## Outputs produced

- `src/splits/make_splits.py`
- `data/processed/splits/*.json`
- `results/tables/split_audit.csv`
- `docs/benchmark_definition.md`
- `results/logs/phase04_splits.log`

## Acceptance check

- Split JSON artifacts exist under `data/processed/splits/`.
- `split_audit.csv` exists and contains:
  - ready `random_split` rows
  - ready stress-split rows
  - explicit unsupported rows
  - `external_holdout` rows for `GSE284197_screen`
  - `dataset_heldout_split` rows with `GSE284197_screen` as a held-out dataset
- Declared holdout-variable leakage is zero for the ready holdout families checked in the audit:
  - unseen perturbation
  - unseen cell state (cell track)
  - unseen lineage (cell track)
  - dataset heldout
  - external holdout
- `docs/benchmark_definition.md` exists and reflects the implemented semantics.
- No Phase 05 baseline training was started.

## Phase result

Phase 04 is complete. The project now has a reproducible benchmark split system and can move into Phase 05 baseline training on the signature track, with the cell track available for auditing and support diagnostics.
