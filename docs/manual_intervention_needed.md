# Manual Intervention Needed

Last synchronized: 2026-09-05 Asia/Shanghai

## Current manual actions

- Review the CBAC single-anonymized submission package after synchronization.
- Confirm the graphical abstract and figure formats against the current CBAC
  upload requirements.
- Install `gears` and `torch_geometric` only if GEARS performance evidence is
  required; the current manuscript deliberately reports GEARS as an
  output-contract demonstration.

The table below retains the original phase-level action history. Current
status is tracked in `docs/project_status.md`.

| Issue | Required user action | URL or file needed | Priority | Status |
|---|---|---|---|---|
| None for Phase 00 | No action needed | N/A | N/A | Closed |
| None for Phase 02 | NormanWeissman2019 download succeeded and checksum matched; no manual download needed | N/A | N/A | Closed |
| None for full scPerturb sync | All 54 official `.h5ad` files are present locally and temporary interrupted-download files were removed | N/A | N/A | Closed |
| None for Phase 03 recovery | The staged preprocessing contract is running locally, `GSE284197_screen` has replaced `E-MTAB-14567` as the active external holdout, and no manual download or metadata rescue is currently required | N/A | N/A | Closed |
| None for Phase 04 | Split generation, audit writing, and benchmark-definition drafting completed locally; no manual intervention is required before Phase 05 baselines | N/A | N/A | Closed |
| Historical Phase 05 | The initial 210-run baseline campaign completed locally; the current expanded campaign is tracked separately in the synchronized status | N/A | N/A | Historical |
| None for Phase 06 | The full evaluation rerun completed locally, required tables were written, and no manual cleanup or external data rescue is required before Phase 07 planning | N/A | N/A | Closed |
| Phase 10 author and submission metadata | Author list, affiliations, funding, competing-interest statement, and target venue choice were provided and incorporated into the Phase 11 title page and package | manuscript/title_page.tex, submit/title_page.pdf | Medium | Closed |
| Current CBAC graphical abstract and figure review | Manually review `submit/graphical_abstract.png` and the active CBAC figure set against the final journal upload requirements | submit/graphical_abstract.png, submit/figures/cbac/, submit/graphical_abstract_note.md | Medium | Open |
| None for post-review figure adjustment 2 | Python/matplotlib five-main-figure rebuild, PTL curve-source output, manuscript recompile, and submission-package refresh completed locally; no additional skill or external service is required for this adjustment | results/figures/, submit/figures/, submit/supplementary/figures/, results/logs/latex/post_adjustment2_qa.json | N/A | Closed |


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:46+08:00
- run_id: random_split__NormanWeissman2019_filtered__seed0__signature__gears
- dataset: NormanWeissman2019_filtered
- split: random_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\random_split__NormanWeissman2019_filtered__seed0__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:46+08:00
- run_id: random_split__NormanWeissman2019_filtered__seed1__signature__gears
- dataset: NormanWeissman2019_filtered
- split: random_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\random_split__NormanWeissman2019_filtered__seed1__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:46+08:00
- run_id: random_split__NormanWeissman2019_filtered__seed2__signature__gears
- dataset: NormanWeissman2019_filtered
- split: random_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\random_split__NormanWeissman2019_filtered__seed2__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:47+08:00
- run_id: unseen_combination_split__NormanWeissman2019_filtered__seed0__signature__gears
- dataset: NormanWeissman2019_filtered
- split: unseen_combination_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_combination_split__NormanWeissman2019_filtered__seed0__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:47+08:00
- run_id: unseen_combination_split__NormanWeissman2019_filtered__seed1__signature__gears
- dataset: NormanWeissman2019_filtered
- split: unseen_combination_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_combination_split__NormanWeissman2019_filtered__seed1__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:47+08:00
- run_id: unseen_combination_split__NormanWeissman2019_filtered__seed2__signature__gears
- dataset: NormanWeissman2019_filtered
- split: unseen_combination_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_combination_split__NormanWeissman2019_filtered__seed2__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:48+08:00
- run_id: unseen_perturbation_split__NormanWeissman2019_filtered__seed0__signature__gears
- dataset: NormanWeissman2019_filtered
- split: unseen_perturbation_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_perturbation_split__NormanWeissman2019_filtered__seed0__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:48+08:00
- run_id: unseen_perturbation_split__NormanWeissman2019_filtered__seed1__signature__gears
- dataset: NormanWeissman2019_filtered
- split: unseen_perturbation_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_perturbation_split__NormanWeissman2019_filtered__seed1__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:48+08:00
- run_id: unseen_perturbation_split__NormanWeissman2019_filtered__seed2__signature__gears
- dataset: NormanWeissman2019_filtered
- split: unseen_perturbation_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_perturbation_split__NormanWeissman2019_filtered__seed2__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:48+08:00
- run_id: random_split__ReplogleWeissman2022_K562_essential__seed0__signature__gears
- dataset: ReplogleWeissman2022_K562_essential
- split: random_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\random_split__ReplogleWeissman2022_K562_essential__seed0__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:49+08:00
- run_id: random_split__ReplogleWeissman2022_K562_essential__seed1__signature__gears
- dataset: ReplogleWeissman2022_K562_essential
- split: random_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\random_split__ReplogleWeissman2022_K562_essential__seed1__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:49+08:00
- run_id: random_split__ReplogleWeissman2022_K562_essential__seed2__signature__gears
- dataset: ReplogleWeissman2022_K562_essential
- split: random_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\random_split__ReplogleWeissman2022_K562_essential__seed2__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:49+08:00
- run_id: unseen_perturbation_split__ReplogleWeissman2022_K562_essential__seed0__signature__gears
- dataset: ReplogleWeissman2022_K562_essential
- split: unseen_perturbation_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_perturbation_split__ReplogleWeissman2022_K562_essential__seed0__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:49+08:00
- run_id: unseen_perturbation_split__ReplogleWeissman2022_K562_essential__seed1__signature__gears
- dataset: ReplogleWeissman2022_K562_essential
- split: unseen_perturbation_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_perturbation_split__ReplogleWeissman2022_K562_essential__seed1__signature__gears.log


## GEARS mandatory model blocked

- Timestamp: 2026-04-26T21:08:50+08:00
- run_id: unseen_perturbation_split__ReplogleWeissman2022_K562_essential__seed2__signature__gears
- dataset: ReplogleWeissman2022_K562_essential
- split: unseen_perturbation_split
- missing_dependencies: torch_geometric, gears
- recommended_action: Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.
- log_file: H:\2026try\4.24\results\logs\training\unseen_perturbation_split__ReplogleWeissman2022_K562_essential__seed2__signature__gears.log


## Phase 11 expanded dataset-heldout baseline block requires scheduled long run or optimization

- Timestamp: 2026-04-27T05:20:00+08:00
- Status: 465 / 525 transparent baseline runs completed; all remaining 60 rows are `dataset_heldout_split`.
- Observation: a grouped `dataset_heldout_split` attempt with three low-cost models for one held-out dataset timed out after 90 minutes on the expanded 9-dataset surface.
- Reason: the all-dataset heldout path repeatedly materializes the expanded multi-dataset signature surface and is no longer suitable for interactive small-batch execution.
- Required action: either schedule the remaining `dataset_heldout_split` rows in a long-running/overnight window, or optimize all-dataset split loading before marking the Opinion 3 full rerun complete.
- Safety note: do not use one-shot `run-all` on the full manifest; use guarded single-row or optimized grouped execution only.

Update 2026-04-27T07:18:00+08:00:

- All-dataset split loading has been optimized and the temporary-manifest overwrite bug in `run_campaign` has been fixed.
- Dataset-heldout is no longer blocked by preprocessing/materialization, but it remains a long-running scheduled compute block.
- Current guarded continuation is running with `src/utils/run_baseline_grouped_batch.py`, one split group at a time, with memory and process guards enabled.
- The GEARS dependency blocker below is still unresolved and remains mandatory for the Opinion 3 model-extension requirement.
