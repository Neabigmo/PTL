# Phase 09 Figures And Visual Design

Date: 2026-04-26

## Scope
- Generate six draft publication-style figures from Phase 08 synthesis outputs.
- Keep the core implementation local and deterministic.
- Write per-figure notes and update project tracking files.

## Skills And Environment
- Used the `scientific-visualization` and `matplotlib` skills for publication-style static figures.
- Read the `browser-use` skill and attempted Browser Use HTML QA for `fig1`, `fig2`, and `fig4`.
- Browser Use was blocked by the local node_repl runtime resolving `D:\heroku-win32-x64\heroku\bin\node.exe` v20.17.0 while requiring >= v22.22.0.
- Did not add new dependencies; the implementation uses existing `matplotlib`, `pandas`, `numpy`, and `PIL`.

## Implementation
- Added `src/figures/make_figures.py`.
- Added `tests/figures/test_make_figures.py`.
- Generated:
  - `fig1_concept.html` / `fig1_concept.png`
  - `fig2_benchmark.html` / `fig2_benchmark.png`
  - `fig3_decay_heatmap.png` / `.pdf` / `.svg`
  - `fig4_ptl_architecture.html` / `fig4_ptl_architecture.png`
  - `fig5_selective_prediction.png` / `.pdf` / `.svg`
  - `fig6_failure_modes.png` / `.pdf` / `.svg`
- Wrote notes under `docs/figure_notes/`.

## QA And Fixes
- Ran local PNG dimension and nonblank pixel checks for all six PNG outputs.
- Visually inspected PNG outputs locally.
- Fixed an overlapping-label issue in `fig2_benchmark.png` by switching to a stable three-column row layout.
- Fixed `fig6_failure_modes.png` by aligning duplicate category labels with explicit y positions and moving the severity legend below the panel.

## Commands
```powershell
E:\anaconda3\envs\sw_mgli\python.exe -m pytest tests\figures -q
E:\anaconda3\envs\sw_mgli\python.exe -m pytest tests\evaluation tests\transportability tests\figures -q
E:\anaconda3\envs\sw_mgli\python.exe src\figures\make_figures.py
```

## Acceptance
- Combined tests passed: `19 passed`.
- Phase 09 CLI completed and wrote `21` files.
- All required PNG outputs exist with nonzero dimensions and nonblank pixel ranges.
- HTML sources exist for `fig1`, `fig2`, and `fig4`; Browser Use visual QA remains pending only because of the local Node runtime mismatch, not because of figure-generation failure.

