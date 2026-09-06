# PTL-v1 provenance anchor

Date: 2026-09-06
Workspace: `H:\2026try\9.5PTL`

This document records the pre-ICLR-refactor evidence surface. It is an
immutability and rollback anchor for the transition, not permission to delete,
rewrite or re-run the v1 work.

## Evidence surface

- 54 raw scPerturb `.h5ad` files under `data/raw/scperturb_v1.4/`.
- Nine processed dataset families under `data/processed/`, including Norman,
  Replogle K562 and RPE1, Adamson 10X005, Datlinger 2017/2021, Dixit,
  GSE284197 and Papalexi.
- `results/tables/baseline_run_matrix.csv` contains 525 run rows plus a header.
- PTL evidence includes `ptl_metrics.csv`, `ptl_ablation.csv`,
  `ptl_feature_audit.csv`, `ptl_examples.parquet`,
  `ptl_oof_predictions.csv`, `reliability_baseline_summary.csv` and the
  selective-prediction summaries.
- The active manuscript is under `manuscript/`; CBAC-facing files are under
  `submit/` and `submission_package/`.
- Figures and LaTeX logs remain under their existing `figures/`, `results/`
  and `manuscript/` paths.

## Known limitations carried forward

1. The current GEARS work is an output-contract compatibility demonstration,
   not a competitive official GEARS benchmark.
2. Historical configs and generated manifests contain old absolute paths such
   as `H:\2026try\4.24`; v2 loaders must never use them.
3. v1 confidence fields mix support, similarity and context heuristics, so
   they are not interchangeable with predictor-generated uncertainty.
4. The v1 PTL split is organized around run-level groups; v2 must use
   `biological_instance_id` grouped cross-fitting.
5. The existing CBAC evidence is retained as provenance and is not silently
   relabeled as ICLR-v2 evidence.

## Why this anchor is necessary

The concrete failure scenario is that refactoring the non-Git workspace while
`_github_ptl_work/` remains a second checkout could overwrite or mix source
states, making it impossible to reproduce which code produced the CBAC
artifacts. A root Git commit and an explicit v1 tag provide a reversible source
anchor. Ordinary tests, type checks, biological primary keys and run metadata
cannot preserve the complete pre-refactor file layout or distinguish two
diverged code trees. A separate per-file hash manifest would be redundant once
the source files are committed and content-addressed by Git, so this bootstrap
uses the Git anchor rather than adding another hash contract.

No raw data, processed data, results, submission packages or manuscript outputs
are moved or deleted during G0.
