# PTL: Reliability-ranking transport across biological contexts

This is the anonymized core release accompanying the ICLR 2027 manuscript
“Measuring the Transport of Perturbation Reliability Rankings Across Biological
Contexts”. It contains the source code, repository-relative configurations,
manuscript sources, publication figures, and compact reviewable tables needed
to inspect the reported analysis.

The release intentionally excludes raw single-cell matrices, cell-level
metadata, model checkpoints, generated caches, local machine paths, remote
execution records, credentials, and internal audit logs. The compact CSV/JSON
products in `artifacts/manifests/` are de-identified summary products for the
reported figures and tables; they are not a redistribution of the underlying
cells.

## Layout

- `src/`: PTL data, prediction, uncertainty, reliability, evaluation, and
  model-contract code.
- `scripts/`: analysis and figure-generation entry points.
- `configs/`: repository-relative data and evaluation contracts.
- `paper/iclr2027/`: anonymous manuscript source and publication figures.
- `artifacts/manifests/`: compact figure and claim summaries.
- `third_party/iclr2027/`: the ICLR style files required to compile the paper.

## Reproduction outline

Install the core Python dependencies with `pip install -e .[core]`. Obtain the
public source data described in `DATA_AVAILABILITY.md`, place them under the
repository-relative paths declared in `configs/`, and then run the relevant
entry points in `scripts/`. The publication PDFs can be compiled from
`paper/iclr2027/` with:

```text
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode supplement.tex
```

The manuscript records the estimand, fixed splits, measurement-depth protocol,
and the boundary between primary evidence and supplementary diagnostics.

The V2.1 supplementary extension is reproducible with the chunked
source-only family measurement helpers under `scripts/`, the corrected
`scripts/run_finite_measurement_simulation_v21.py --mode primary` grid,
`scripts/merge_finite_measurement_v21.py`, and
`scripts/figures/supp_upgrade_v2.py`. The release also includes the strict
five-fold source-frozen GEARS bridge, calibrated finite-measurement inference,
the compact predictor-family figure, and canonical five-family CSV/JSON
summaries. Large prediction tensors, bootstrap inputs, model checkpoints, and
raw matrices remain excluded from this anonymized release.

## Code and data availability

The canonical anonymous release is the `iclr2027-anonymous-core` branch of
the PTL repository:

https://github.com/Neabigmo/PTL/tree/iclr2027-anonymous-core

The Frangieh RNA screen and the Nadig HepG2/Jurkat screens are available from
the public scPerturb v1.4 Zenodo record. See `DATA_AVAILABILITY.md` for the
record DOI and the exact file names used by the analysis. Users should obtain
the data directly and follow the original dataset terms.
