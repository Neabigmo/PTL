# PTL: ICLR 2027 reliability project

This workspace contains two explicitly separated lines:

- **Active PTL-v2 / ICLR 2027:** the new reliability-calibration study under
  `src/ptl/`, `configs/`, `artifacts/`, and `paper/iclr2027/`.
- **Legacy PTL-v1 / CBAC:** the preserved prior analysis and submission line
  under `src/analysis/`, `src/baselines/`, `src/transportability/`,
  `manuscript/`, `results/`, and the provenance anchor documented in
  `docs/V1_PROVENANCE_FREEZE.md`.

The ICLR line asks how much reliability information remains in biological
context after predictor uncertainty is known. Its design contract is
[`docs/ICLR2027_REFACTOR_MASTERPLAN.md`](docs/ICLR2027_REFACTOR_MASTERPLAN.md),
with the current accelerated execution packet in
[`docs/CODEX_NEXT_SPRINT_G1_G4.md`](docs/CODEX_NEXT_SPRINT_G1_G4.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/` | PTL-v2 dataset, environment, predictor, task, reliability and dependency contracts. |
| `src/ptl/` | Active PTL-v2 data, predictor, UQ, reliability, evaluation and experiment modules. |
| `artifacts/` | Tracked compact manifests/source-data contracts; large local artifacts stay ignored. |
| `paper/iclr2027/` | Separate ICLR 2027 paper surface; it does not overwrite the CBAC manuscript. |
| `src/analysis/`, `src/baselines/`, `src/transportability/` | Legacy v1/CBAC implementation retained for provenance and supplementary evidence. |
| `data/`, `results/`, `submit/`, `submission_package/` | Existing local data and generated outputs; ignored by the active Git source tree. |
| `docs/` | Design contract, provenance freeze, execution gates and audit records. |
| `third_party/iclr2027/` | User-provided ICLR 2027 LaTeX style files. |
| `_github_ptl_work/` | Historical Git worktree retained as read-only legacy context, not an active source tree. |

## Rules for the active line

- Do not use legacy heuristic `confidence` as headline model-native UQ; label it
  `legacy_confidence_proxy` when it is retained for comparison.
- Keep `predictions`, `deployment_features`, `outcomes`, and `folds` separate;
  outcome-derived fidelity cannot enter deployment features.
- Group PTL cross-fitting by `biological_instance_id`, not by run ID.
- Report environment-level metrics before macro-averaging so large screens do
  not dominate the conclusions.
- Do not run new G1-G4 experiments until the corresponding contracts and
  manifests are present and tested.

## Legacy snapshot

The former CBAC line remains available locally, including its nine processed
screen families, baseline matrices, manuscript source, figures, and submission
records. It is not silently rewritten by the ICLR-v2 refactor.
