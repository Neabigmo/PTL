# Perturbation Transportability Project

This repository contains the data processing, modeling, evaluation, and manuscript materials for a single-cell perturbation transportability study. The workspace is organized so that raw inputs, reproducible code, analysis outputs, manuscript assets, and submission materials remain clearly separated.

## Project Layout

| Path | Purpose |
| --- | --- |
| `config/` | Dataset and preprocessing configuration files. |
| `data/` | External, raw, interim, and processed data products. |
| `src/` | Analysis, model training, evaluation, figure, and utility code. |
| `tests/` | Regression and smoke tests for the analysis code. |
| `results/` | Generated figures, tables, logs, models, checkpoints, and QA outputs. |
| `manuscript/` | LaTeX manuscript source, compiled PDFs, figures, tables, and revision notes. |
| `submit/` | Current journal-facing submission files. |
| `submission_package/` | Packaged submission materials and release archive. |
| `docs/` | Study documentation, data inventory, decision records, and internal notes. |
| `scripts/maintenance/` | Environment checks, manuscript verification, and maintenance utilities. |
| `third_party/` | External repositories or vendored research dependencies. |
| `tools/` | Local support modules and project-specific tooling. |
| `.cache/` | Local test caches and temporary working files. |

## Working Conventions

- Keep source data under `data/external/` or `data/raw/`; derived tables and matrices belong under `data/processed/`.
- Keep reusable analysis logic in `src/`; one-off maintenance checks belong under `scripts/maintenance/`.
- Write generated research outputs to `results/`, then copy publication-ready material to `manuscript/` or `submit/`.
- Keep revision notes, reviewer-response drafts, and manuscript adjustment records under `manuscript/revision_notes/`.
- Treat `docs/internal_archive/` as provenance material: useful for traceability, but not part of the public-facing project surface.

## Current synchronized snapshot

The active manuscript line is the single-anonymized Computational Biology and
Chemistry (CBAC) version. It is backed by nine processed public screens: eight
scPerturb datasets plus the independent `GSE284197_screen` external holdout.

The current evidence surface contains 525 planned transparent baseline runs
across five baselines and six split families, together with PTL reliability,
public-screen holdout, case-example, and GEARS output-contract artifacts.
The current CBAC figure family is under `results/figures_cbac/`, and the
manuscript source is under `manuscript/`.

The main submission-facing files are in `submit/`. Historical EAAI drafts and
earlier figure families remain in the workspace for provenance, but are not the
active submission line.
