# Project Status

Last synchronized: 2026-09-05 Asia/Shanghai

## Current release line

The active local release line is the single-anonymized Computational Biology
and Chemistry (CBAC) manuscript and submission package. The latest manuscript
source was completed on 2026-05-27 from the expanded public-screen evidence
surface; the CBAC figure family and submission copies were resynchronized on
2026-09-05.

## Current evidence snapshot

- Nine datasets are processed: eight scPerturb datasets plus the independent
  `GSE284197_screen` external holdout.
- `results/tables/baseline_run_matrix.csv` contains 525 planned transparent
  baseline runs across five models, three seeds, and six split families.
- The latest deduplicated registry coverage for those 525 manifest rows is 495
  `success` and 30 `cached`; no manifest row is currently missing from the
  latest aggregate evidence table.
- The latest aggregate evidence is in `results/tables/all_metrics.csv` and the
  derived CBAC summaries are in `results/tables/`.
- The active figure source/output family is `results/figures_cbac/`; the active
  manuscript entry point is `manuscript/main.tex`.

## Active blockers and manual checks

- GEARS has a compatible adapter and three completed demonstration rows, but
  18 planned adapter rows remain missing because the required `gears` and
  `torch_geometric` dependencies are not installed in the preferred
  environment. GEARS is reported as output-contract validation, not as a
  completed competitive benchmark.
- `GSE264667/NadigOConner2024` remains an audited external candidate only; it is
  not included as completed validation evidence.
- The graphical abstract and final CBAC upload requirements still require
  venue-specific manual review.

Historical phase summaries below are retained for provenance. They describe
the state at the time of each phase and should not override the current
evidence snapshot above.

## Historical phase ledger

## Completed phases
- Phase 00: completed on 2026-04-24.
- Phase 01: completed on 2026-04-24.
- Phase 02: completed on 2026-04-24.
- Phase 03: completed on 2026-04-25.
- Phase 04: completed on 2026-04-25.
- Phase 05: completed on 2026-04-26.
- Phase 06: completed on 2026-04-26.
- Phase 07: completed on 2026-04-26.
- Phase 08: completed on 2026-04-26.
- Phase 09: completed on 2026-04-26.
- Phase 10: completed on 2026-04-26.
- Phase 11: completed on 2026-04-26.

## Historical blockers and notes
- No blocker for the staged preprocessing contract on the current pilot datasets.
- `E-MTAB-14567` remains archived as an inspected but non-adopted candidate because per-cell perturbation mapping is not reproducibly exposed by the downloaded bundle.
- The active external validation dataset is now `GSE284197_screen`, which has recoverable per-cell perturbation labels, controls, and reference context fields.
- Zotero MCP was mentioned by the user and checked through tool discovery, but no callable Zotero methods were exposed in this session. Scite was sufficient for Phase 01.
- Preferred environment `E:\anaconda3\envs\sw_mgli` is active and now supports both preprocessing and split generation (`anndata`, `scanpy`, `pyarrow`, `scikit-learn`, `h5py`).
- No active blocker remains for the completed Phase 06 evaluation layer or Phase 07 PTL layer.
- The weakest carry-forward pattern is still cross-dataset / external ridge performance, especially on `GSE284197_screen`; this is now a measured Phase 06 result to analyze in PTL work, not an execution blocker.
- Phase 06 confirmed that transfer-family behavior diverges from within-dataset behavior:
  - ridge leads in `random_split`, `low_support_split`, and `unseen_perturbation_split`
  - `global_delta_baseline` leads in `dataset_heldout_split` and `external_holdout`
- Phase 06 result tables and confidence/transportability outputs are now available for Phase 07 design.
- Phase 07 confirmed that PTL strongly improves false-transportability filtering over naive confidence, while cosine-risk selective ranking remains conservative and should be interpreted separately in Phase 08.
- Phase 08 synthesized the benchmark, PTL, and failure-mode outputs into figure-ready tables and a bounded narrative.
- Phase 09 generated six publication-style draft figures, figure notes, and vector exports for matplotlib figures.
- Phase 10 generated a full manuscript draft scaffold, abstract, outline, references, limitations, and target-venue cover-letter drafts.
- Phase 11 compiled the anonymized manuscript and title page, prepared the EAAI double-anonymized submission package, and completed PDF/submission QA.

## Next action

The CBAC submission package has now been refreshed from the synchronized
manuscript source. Remaining work is limited to final venue-specific manual
checks and, only if required, installing the optional GEARS dependencies. Do
not interpret the historical EAAI package as the current submission artifact.

## Plugin plan
- Plugin usage has been added to [docs/plugin_usage_plan.md](H:/2026try/9.5PTL/docs/plugin_usage_plan.md).
- Current project-relevant plugin: `Browser Use`.
- User-reported Life Science Research plugin capabilities have been added to the plan as conditional phase support.
- Project-relevant skills/tools have also been mapped by phase: literature, data handling, single-cell preprocessing, baselines, evaluation, PTL, figures, manuscript, and PDF/submission checks.
- Planned Browser Use phases: Phase 09 (visual QA), Phase 11 (submission artifact inspection).

## Token-saving protocol
- Use only PROJECT_RULES.md, SKILLS_INDEX.md, the current task packet, and required status files during execution.
- For every project-level task, check SKILLS_INDEX.md during planning. If a relevant skill is found, stop and ask the user to add that skill in the next conversation.
- Save detailed notes to docs/task_logs/ and results/logs/.
- Summarize outcomes here after each phase.

## Phase 00 summary
- Created required project directory structure under the current project root.
- Created 12 concise task packets in docs/task_packets/.
- Initialized project tracking files.
- Added plugin/tool usage plan and phase-specific tool notes.
- Added Life Science Research plugin capability mapping from user-provided skill list.
- Added SKILLS_INDEX.md skill gate rule for all future project-level task planning.
- No data downloads, model training, experiments, or manuscript drafting were started.

## Phase 01 summary
- Used literature-review workflow and Scite literature MCP to build a verified literature map.
- Created docs/literature_map.md with 44 papers/resources and one-line relevance statements.
- Created docs/research_gap.md with a bounded gap centered on perturbation transportability, context-stress benchmarks, reliability calibration, and failure-mode analysis.
- Created docs/positioning_statement.md with five manuscript contributions and target-journal angles for EAAI, KBS, AIME, JBI, and CMPB.
- Recorded detailed searches and acceptance checks in docs/task_logs/phase01_literature_and_positioning.md.
- No Phase 02 data downloads or experiments were started.

## Phase 02 summary
- Checked the skill gate and read only the relevant data-skill documentation before execution.
- Used the official scPerturb v1.4 Zenodo record as the primary data source.
- Downloaded all 54 official `.h5ad` files from the scPerturb v1.4 Zenodo record into `data/raw/scperturb_v1.4/`.
- Retried the Norman download with proxy at the user's request; final file size and MD5 match Zenodo metadata.
- Synced the local directory against the Zenodo record and confirmed 54 / 54 `.h5ad` files are present with size match.
- Removed stale `.tmp` artifacts from interrupted downloads after completion checks.
- Rebuilt `results/tables/data_inventory.csv` and updated `docs/data_inventory.md` to reflect full acquisition.
- Recorded download logs in `results/logs/downloads/` and detailed execution notes in `docs/task_logs/phase02_data_acquisition.md`.
- No Phase 03 preprocessing, benchmark splits, model training, or evaluation were started.

## Phase 03 recovery summary
- Installed the minimum preprocessing stack into `E:\anaconda3\envs\sw_mgli`.
- Reworked `src/data/preprocess_dataset.py` into a staged low-memory pipeline with cell metadata, QC manifests, feature metadata, matrix-store manifests, pseudobulk tables, and delta signature tables.
- Reprocessed `NormanWeissman2019_filtered` under the new contract and wrote a cell-level convenience AnnData only as an optional artifact.
- Reprocessed `ReplogleWeissman2022_K562_essential` under the same contract without forcing a dense filtered-matrix rewrite.
- Archived `E-MTAB-14567` as an inspected but non-adopted external candidate.
- Downloaded, resumed, inspected, and adopted `GSE284197_screen` from GEO as the active external validation dataset.
- Processed `GSE284197_screen` into the same staged artifact family as the internal pilot datasets.
- Rewrote `results/tables/preprocessing_summary.csv` to reflect the staged contract instead of the earlier pilot-only schema.

## Phase 04 summary
- Used the Phase 03 recovery artifacts only and implemented `src/splits/make_splits.py` as the first split-generation CLI.
- Wrote deterministic dual-track split manifests to `data/processed/splits/` for seeds `0`, `1`, and `2`.
- Created `results/tables/split_audit.csv` with leakage diagnostics, support counts, and signature-track distance diagnostics.
- Created `docs/benchmark_definition.md` as the first formal benchmark specification for the project.
- Kept `GSE284197_screen` in two roles simultaneously:
  - dedicated `external_holdout`
  - symmetric member of `dataset_heldout_split`
- Ready families now include:
  - `random_split`
  - `unseen_perturbation_split`
  - `unseen_combination_split` for Norman only
  - `low_support_split`
  - `dataset_heldout_split`
  - `external_holdout`
  - `unseen_cell_state_split` on the `GSE284197_screen` cell track
  - `unseen_cell_type_or_lineage_split` on the `GSE284197_screen` cell track
- Explicit unsupported records were kept for infeasible families instead of silently dropping them, especially signature-track state/lineage stress and non-existent combination regimes in Replogle/GSE.

## Phase 05 summary
- Implemented `src/baselines/run_baseline.py` as the single Phase 05 CLI for run-matrix generation, single-run execution, full-campaign execution, and summary rebuilding.
- Kept the Phase 05 core to the 5 agreed full-gene baselines:
  - `control_mean_baseline`
  - `global_delta_baseline`
  - `perturbation_mean_delta_baseline`
  - `cell_context_knn_delta_baseline`
  - `ridge_regression_baseline`
- Built `results/tables/baseline_run_matrix.csv` with `210` planned runs from the ready signature split manifests.
- Completed all `210` manifest runs with no failed latest status:
  - `196` latest `success`
  - `14` latest `cached`
- Wrote per-run bundles under `results/baselines/`, appended registry entries to `results/tables/experiment_registry.csv`, and produced:
  - `results/tables/baseline_metrics_by_split.csv`
  - `results/tables/model_ranking_instability.csv`
  - `results/tables/phase05_global_gene_intersection_7880.txt`
- Preserved the hybrid gene-space contract:
  - native genes for within-dataset splits
  - fixed 7,880-gene global intersection for `dataset_heldout_split` and `external_holdout`
- Overall mean-rank ordering favored:
  1. `perturbation_mean_delta_baseline`
  2. `ridge_regression_baseline`
  3. `global_delta_baseline`
  4. `cell_context_knn_delta_baseline`
  5. `control_mean_baseline`
- Cross-dataset and external-transfer regimes were harder than within-dataset random / low-support settings, and ridge weakened notably on the active external holdout `GSE284197_screen`.

## Phase 06 summary
- Implemented and completed the rigorous evaluation layer in `src/evaluation/`.
- Used `baseline_run_matrix.csv` as the canonical universe and latest-status deduplication from `experiment_registry.csv` to build the formal selection audit.
- Evaluated all `210` formal Phase 05 signature-track runs and excluded the `5` non-manifest registry extras / smoke runs.
- Wrote:
  - `results/tables/run_selection_audit.csv`
  - `results/tables/per_signature_metrics.parquet`
  - `results/tables/all_metrics.csv`
  - `results/tables/phase06_summary_by_family.csv`
  - `results/tables/model_pairwise_comparisons.csv`
  - `docs/metric_dictionary.md`
- Re-ran evaluation tests and confirmed `6 passed`.
- Overall mean `mean_cosine_non_control` ranking across all runs is now:
  1. `ridge_regression_baseline`
  2. `perturbation_mean_delta_baseline`
  3. `cell_context_knn_delta_baseline`
  4. `global_delta_baseline`
  5. `control_mean_baseline`
- Family-level winners differ by stress regime:
  - `ridge_regression_baseline` leads `random_split`, `low_support_split`, and `unseen_perturbation_split`
  - `perturbation_mean_delta_baseline` leads `unseen_combination_split`
  - `global_delta_baseline` leads `dataset_heldout_split` and `external_holdout`

## Phase 07 summary
- Implemented the PTL reliability layer in `src/transportability/`.
- Built `127,135` non-control stress-signature PTL examples from Phase 06 outputs across `165` runs.
- Wrote:
  - `results/tables/ptl_examples.parquet`
  - `results/tables/ptl_metrics.csv`
  - `results/tables/ptl_ablation.csv`
  - `results/tables/failure_mode_summary.csv`
  - `results/tables/ptl_run_summary.json`
  - `results/logs/training/phase07_ptl.log`
- Added focused PTL tests under `tests/transportability/`.
- Re-ran combined tests and confirmed `10 passed`.
- Best completed PTL ablation by false-transportability filtering:
  - `no_context_distance` / `random_forest`
  - ROC AUC `0.907288`
  - average precision `0.922685`
  - mean false-transportability rate improved from naive `0.504201` to `0.173978`
  - gain versus naive confidence: `0.330222`
- Optional biological-support/pathway features were left as a skipped ablation because no standalone external annotation client is available inside the Phase 07 CLI; the core PTL remains fully local and reproducible.

## Phase 08 summary
- Implemented `src/evaluation/main_analysis.py` as the Phase 08 synthesis CLI.
- Wrote:
  - `results/tables/main_findings.csv`
  - `results/tables/transfer_decay_summary.csv`
  - `results/tables/selective_prediction_summary.csv`
  - `results/tables/failure_mode_atlas.csv`
  - `docs/results_narrative.md`
  - `results/logs/phase08_main_analysis.log`
- Added Phase 08 tests in `tests/evaluation/test_main_analysis.py`.
- Re-ran combined tests and confirmed `14 passed`.
- Main synthesized findings:
  - `external_holdout` is the hardest stress family by average non-control cosine.
  - Random-split winner `ridge_regression_baseline` does not remain best in every stress family.
  - `global_delta_baseline` leads `dataset_heldout_split` and `external_holdout`.
  - PTL improves false-transportability filtering over naive confidence; best row remains `no_context_distance` / `random_forest`.
  - Failure modes concentrate in novelty, low-support, and transfer-boundary settings.

## Phase 09 summary
- Implemented `src/figures/make_figures.py` as the Phase 09 deterministic figure-generation CLI.
- Wrote:
  - `results/figures/fig1_concept.html` and `.png`
  - `results/figures/fig2_benchmark.html` and `.png`
  - `results/figures/fig3_decay_heatmap.png`, `.pdf`, and `.svg`
  - `results/figures/fig4_ptl_architecture.html` and `.png`
  - `results/figures/fig5_selective_prediction.png`, `.pdf`, and `.svg`
  - `results/figures/fig6_failure_modes.png`, `.pdf`, and `.svg`
  - `docs/figure_notes/fig1_concept.md` through `fig6_failure_modes.md`
  - `results/logs/figures/phase09_figures.log`
- Added focused figure-generation tests in `tests/figures/test_make_figures.py`.
- Re-ran combined tests and confirmed `19 passed`.
- Performed local PNG dimension/nonblank checks for all six PNG outputs and local visual inspection; Browser Use HTML QA was attempted but blocked by the node_repl Node runtime resolving to v20.17.0 while requiring >= v22.22.0.

## Phase 10 summary
- Used scientific-writing, citation-management, and venue-template skills plus Scite metadata checks for the key manuscript citation backbone.
- Wrote:
  - `docs/manuscript_outline.md`
  - `docs/abstract_v1.md`
  - `manuscript/main.tex`
  - `manuscript/sections/introduction.tex`
  - `manuscript/sections/methods.tex`
  - `manuscript/sections/results.tex`
  - `manuscript/sections/discussion.tex`
  - `manuscript/sections/limitations.tex`
  - `manuscript/references.bib`
  - `docs/cover_letter_EAAI.md`
  - `docs/cover_letter_KBS.md`
  - `docs/cover_letter_AIME.md`
  - `docs/task_logs/phase10_manuscript_writing.md`
- Referenced all six Phase 09 figures and the core Phase 08 evidence tables in the manuscript draft.
- Kept claims bounded to benchmarked single-cell perturbation signatures, context-stress evaluation, PTL false-transportability filtering, and descriptive failure-mode analysis.
- Lightweight QA confirmed required files, figure references, citation keys, and bounded-claim phrase checks before Phase 11.

## Phase 11 summary
- Prepared EAAI double-anonymized submission materials:
  - `submit/manuscript.pdf`
  - `submit/title_page.pdf`
  - `submit/cover_letter.md`
  - `submit/highlights.md`
  - `submit/data_code_availability.md`
  - `submit/graphical_abstract.png`
  - `submit/graphical_abstract_note.md`
  - `submit/package_manifest.md`
  - `submit/figures/`
  - `submit/tables/`
- Added `manuscript/title_page.tex` and kept `manuscript/main.tex` anonymized.
- Compiled with TeX Live 2025:
  - `results/logs/latex/main_compile.log`
  - `results/logs/latex/title_page_compile.log`
- Ran PDF QA:
  - rendered 10 anonymized manuscript pages and 1 title page using `pdftoppm`
  - confirmed rendered pages were nonblank
  - confirmed no exact author identity phrases in anonymized manuscript text
  - confirmed no forbidden project claims in submission-facing text
  - confirmed graphical abstract size `1328 x 531`
- Wrote `docs/task_logs/phase11_latex_submission_package.md` and `results/logs/latex/phase11_pdf_qa.json`.
- Applied `投稿前调整意见1.txt` as a post-submission-package refinement:
  - regenerated the figure set as denser multi-panel evidence figures:
    `fig1_context_stress_workflow`, `fig2_benchmark_audit`,
    `fig3_model_ranking_instability`, `fig4_ptl_reliability`,
    `fig5_failure_mode_atlas`, and `fig6_robustness_audit`
  - updated `manuscript/sections/results.tex` to reference the refined figure sequence
  - recompiled `manuscript/main.pdf`; the post-adjustment compile log reports 11 pages and no fatal LaTeX errors
  - refreshed `submit/manuscript.pdf` and `submit/figures/` with the refined figure set
  - reran figure tests and confirmed `5 passed`

## Post-submission adjustment 2 summary
- Applied `投稿前调整意见2.txt` by replacing the prior HTML-led figure sequence with a Python/matplotlib-first five-main-figure system.
- Rebuilt `src/figures/make_figures.py` around pure-white journal-style figures, fixed semantic palette constants, smaller panel typography, and PNG/PDF/SVG exports.
- Added Phase 07 per-example PTL prediction output:
  - `results/tables/ptl_oof_predictions.csv`
  - used by Figure 4 ROC, precision--recall, and risk--coverage panels.
- New main figure sequence:
  - `fig1_study_design`
  - `fig2_benchmark_composition_audit`
  - `fig3_ranking_instability_transfer_decay`
  - `fig4_ptl_selective_filtering`
  - `fig5_failure_mode_atlas`
- Moved the robustness/audit view to `supp_fig1_robustness_audit` and copied it under `submit/supplementary/figures/`.
- Updated `manuscript/sections/results.tex` to cite five main figures only, with robustness/audit handled as supplementary support.
- Recompiled the anonymized manuscript with TeX Live `latexmk`; refreshed `submit/manuscript.pdf`.
- Verification completed:
  - `tests/evaluation tests/transportability tests/figures`: `19 passed`
  - figure smoke tests after warning cleanup: `5 passed`
  - `src/figures/make_figures.py` completed without warnings
  - PDF text anonymity scan found no exact author identity phrases
  - forbidden-claim scan found no hits
  - rendered manuscript pages 1, 5, 7, 8, and 9 were nonblank

## Post-submission adjustment 3 / major reinforcement status
- In progress under `docs/task_logs/phase11_major_reinforcement_opinion3.md`.
- Added and processed six additional local scPerturb datasets, bringing the processed benchmark surface to 8 scPerturb datasets plus one external holdout.
- Rebuilt split artifacts after reference and combination-label fixes; `DatlingerBock2021` now contributes non-control signatures, and false combinatorial splits from guide/library suffixes are marked unsupported rather than run.
- Added GEARS-compatible adapter and blocker logging. GEARS remains mandatory but blocked because `torch_geometric` and `gears` are absent from `E:\anaconda3\envs\sw_mgli`; blocker rows/logs were written and this cannot be marked submission-complete until resolved.
- Added pathway/retrieval metrics, deployment-only PTL feature audit, and split difficulty synthesis support.
- Rebuilt baseline run matrix to 525 rows. One-shot `run-all` caused workstation resource pressure, so execution was switched to guarded small-batch execution.
- Safe batch progress: 465 / 525 baseline runs completed. All non-`dataset_heldout_split` rows are complete; the remaining 60 dataset-heldout rows require a scheduled long run or loading optimization.
