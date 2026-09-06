# Phase 07 PTL Model

Last updated: 2026-04-26 12:16 Asia/Shanghai

## Scope
- Implemented PTL as a model-agnostic reliability layer over Phase 06 signature-track stress outputs.
- Kept PTL distinct from perturbation-response prediction.
- Used only local Phase 03-06 artifacts for the core run.
- Optional Life Science Research pathway/network features were recorded as unavailable for the standalone CLI and kept as a skipped ablation hook.

## Skills and environment
- Checked `SKILLS_INDEX.md` before execution.
- Used the `scikit-learn` skill for classical ML pipeline design.
- Executed in `E:\anaconda3\envs\sw_mgli`.
- The project root is not a git repository, so validation used file outputs and tests rather than git status.

## Implementation summary
- Added `src/transportability/ptl.py` with:
  - PTL example construction from `per_signature_metrics.parquet`, `all_metrics.csv`, the Phase 05 run manifest, split audit, and prepared split data.
  - Leakage-aware feature inference that excludes observed outcome columns such as cosine, risk, transportable labels, and true delta norm.
  - Train-only support and novelty features for perturbations, components, references, datasets, batch, and timepoint.
  - Split-audit context features including train-test centroid distance and overlap diagnostics.
  - Logistic regression and random forest classifier evaluation under PTL ablations.
- Added `src/transportability/train_ptl.py` as the Phase 07 CLI.
- Added `tests/transportability/test_ptl.py`.

## Execution summary
- Ran combined tests:
  - `python -m pytest tests/evaluation tests/transportability -q`
  - Result: `10 passed`
- Ran Phase 07 smoke on real Phase 06 outputs:
  - `python src/transportability/train_ptl.py --max-examples 2000 --output-dir results/_phase07_smoke --registry results/tables/_phase07_smoke_registry.csv --log-file results/logs/training/phase07_ptl_smoke.log --run-label smoke_phase07_ptl`
  - Result: completed and wrote smoke PTL tables.
- Ran full Phase 07:
  - `python src/transportability/train_ptl.py --output-dir results --registry results/tables/experiment_registry.csv --log-file results/logs/training/phase07_ptl.log --run-label phase07_ptl`
  - Result: completed and appended `8` Phase 07 rows to `experiment_registry.csv`.

## Outputs produced
- `src/transportability/__init__.py`
- `src/transportability/ptl.py`
- `src/transportability/train_ptl.py`
- `tests/transportability/test_ptl.py`
- `results/tables/ptl_examples.parquet`
- `results/tables/ptl_metrics.csv`
- `results/tables/ptl_ablation.csv`
- `results/tables/failure_mode_summary.csv`
- `results/tables/ptl_run_summary.json`
- `results/logs/training/phase07_ptl.log`
- `results/_phase07_smoke/`
- `results/logs/training/phase07_ptl_smoke.log`

## Acceptance summary
- PTL examples: `127,135`
- Unique stress runs represented: `165`
- Transportable rate: `0.498100`
- Best completed ablation by false-transportability filtering:
  - `no_context_distance` / `random_forest`
  - ROC AUC: `0.907288`
  - average precision: `0.922685`
  - Brier score: `0.126724`
  - mean false-transportability rate: `0.173978`
  - naive confidence mean false-transportability rate: `0.504201`
  - gain versus naive confidence: `0.330222`
- The full PTL random forest was close:
  - mean false-transportability rate: `0.174228`
  - gain versus naive confidence: `0.329973`

## Interpretation notes
- PTL strongly improves detection/filtering of non-transportable outputs relative to naive confidence.
- Cosine-risk selective ranking is more conservative: PTL lowers false transportability but does not lower mean cosine-derived selective risk versus naive confidence in this run.
- Phase 08 should report both views clearly:
  - PTL succeeds on its direct transportability target.
  - Selective-risk curves based on raw cosine risk are a secondary diagnostic, not the primary PTL acceptance criterion.

## Acceptance checklist
- [x] PTL code exists.
- [x] PTL metrics table exists.
- [x] PTL ablation table exists.
- [x] Failure-mode summary exists.
- [x] Training log exists.
- [x] Registry rows appended.
- [x] Combined evaluation and PTL tests pass.
- [x] Tracking files updated.
