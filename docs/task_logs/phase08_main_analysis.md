# Phase 08 Main Analysis

Last updated: 2026-04-26 12:39 Asia/Shanghai

## Scope
- Synthesized completed Phase 05-07 outputs into figure-ready analysis tables and a bounded results narrative.
- Did not create figures; Phase 09 owns visual design and rendered figures.
- Did not call external biological databases; the existing local outputs were sufficient for Phase 08 acceptance.

## Skills and environment
- User supplied the `scikit-learn` skill for this phase.
- Checked `SKILLS_INDEX.md`; no additional skill was required for the local synthesis task.
- Executed in `E:\anaconda3\envs\sw_mgli`.

## Implementation summary
- Added `src/evaluation/main_analysis.py`.
- Implemented:
  - transfer decay and split-family ranking synthesis
  - PTL versus naive confidence selective prediction synthesis
  - failure-mode atlas aggregation
  - compact main findings table
  - bounded Markdown narrative generation
- Added `tests/evaluation/test_main_analysis.py`.

## Execution summary
- Ran combined tests:
  - `python -m pytest tests/evaluation tests/transportability -q`
  - Result: `14 passed`
- Ran full Phase 08:
  - `python src/evaluation/main_analysis.py --metrics results/tables/all_metrics.csv`
  - Result: completed and wrote all required outputs.

## Outputs produced
- `src/evaluation/main_analysis.py`
- `tests/evaluation/test_main_analysis.py`
- `results/tables/main_findings.csv`
- `results/tables/transfer_decay_summary.csv`
- `results/tables/selective_prediction_summary.csv`
- `results/tables/failure_mode_atlas.csv`
- `docs/results_narrative.md`
- `results/logs/phase08_main_analysis.log`

## Main synthesized findings
- `external_holdout` is the hardest stress family by average non-control cosine.
- Random-split winner `ridge_regression_baseline` does not remain the winner in every stress family.
- `global_delta_baseline` leads `dataset_heldout_split` and `external_holdout`.
- PTL improves false-transportability filtering over naive confidence; best row is `no_context_distance` / `random_forest`.
- The cosine-risk selective view remains separate and more conservative.
- Failure modes concentrate in novelty, low-support, and transfer-boundary settings.

## Acceptance checklist
- [x] `main_findings.csv` exists and is non-empty.
- [x] `transfer_decay_summary.csv` exists and is non-empty.
- [x] `selective_prediction_summary.csv` exists and is non-empty.
- [x] `failure_mode_atlas.csv` exists and is non-empty.
- [x] `docs/results_narrative.md` exists and includes limitations.
- [x] `results/logs/phase08_main_analysis.log` exists.
- [x] Combined tests pass.
- [x] Tracking files updated.
