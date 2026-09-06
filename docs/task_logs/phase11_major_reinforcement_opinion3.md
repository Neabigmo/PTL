# Phase 11 Major Reinforcement From Adjustment Opinion 3

Started: 2026-04-26

## Scope

This task implements `投稿前调整意见3.txt` without reducing the requested scope. The manuscript is being upgraded from a benchmark-style report into a virtual-cell perturbation transportability reliability paper.

## Completed So Far

- Added six additional local scPerturb preprocessing configs and processed them successfully:
  - `AdamsonWeissman2016_GSM2406677_10X005`
  - `DatlingerBock2017`
  - `DatlingerBock2021`
  - `DixitRegev2016_K562_TFs_7_days`
  - `PapalexiSatija2021_eccite_RNA`
  - `ReplogleWeissman2022_rpe1`
- Rebuilt split artifacts from the expanded preprocessing summary:
  - 9 processed datasets total, including 8 scPerturb datasets plus the external holdout.
  - `results/tables/split_audit.csv` now records 231 ready split artifacts and 153 unsupported records.
- Added pathway and perturbation-discrimination metric functions plus tests:
  - `src/evaluation/metrics.py`
  - `src/evaluation/biological_metrics.py`
  - `data/external/gene_sets/transportability_gene_sets.json`
- Added deployment-only PTL feature audit support:
  - `results/tables/ptl_feature_audit.csv` is written when `--write-feature-audit` is used.
  - PTL default feature mode is now `deployment`, excluding target/evaluation/identifier fields.
- Added GEARS-compatible adapter:
  - `src/models/run_perturbation_model.py`
  - `results/tables/perturbation_model_run_matrix.csv`
  - GEARS is currently blocked because `torch_geometric` and `gears` are not installed in `E:\anaconda3\envs\sw_mgli`.
  - 15 GEARS registry/blocker rows and per-run logs were generated.
- Fixed baseline gene-column resolution:
  - `src/baselines/run_baseline.py` now resolves gene columns through each dataset feature metadata rather than assuming every non-fixed metadata column is numeric.
- Rebuilt baseline manifest:
  - `results/tables/baseline_run_matrix.csv`
  - 525 planned transparent baseline runs.

## Resource-Safe Execution Change

The old one-shot `run-all` strategy caused resource pressure during expanded all-dataset heldout evaluation. A resumable batch runner was added:

- `src/utils/run_baseline_batch.py`

The runner:

- checks for project-related Python processes before every run;
- optionally clears stale project Python processes with `--kill-zombies`;
- runs each manifest row through an independent `run-one` subprocess;
- releases memory between rows;
- supports small-first scheduling and model/split/dataset filters;
- writes cumulative logs to `results/logs/training/phase11_baseline_batch.log`.

Validated batches:

- 3 low-cost baseline runs completed.
- 10 additional low-cost baseline runs completed.
- 5 kNN baseline runs completed.
- 1 ridge baseline pressure-probe run completed.
- No project Python processes remained after these batches.

Current baseline progress after resource-safe continuation:

- Completed: 465 / 525 baseline runs.
- Pending: 60 / 525 baseline runs.
- All non-`dataset_heldout_split` transparent baseline runs are complete.
- The remaining 60 runs are exactly `dataset_heldout_split`: 12 per baseline model.
- `dataset_heldout_split` is now treated as a long-running scheduled block, not an interactive small task. A grouped one-split attempt for low-cost models timed out after 90 minutes on the expanded 9-dataset surface.

Additional fixes made during continuation:

- `DatlingerBock2021` delta signatures were repaired by changing the reference stratum from `sample` to `perturbation_2`; non-control signatures increased from 0 to 79.
- Combination split parsing now excludes guide/plasmid/library tokens such as `only`, `pMJ*`, numeric guide suffixes, and `Tcrlibrary`, preventing single-guide libraries from being mislabeled as combinatorial perturbations.
- The baseline batch runner now treats a run as complete only when `run_metrics.json` exists.
- Process and memory guards were made less fragile by replacing PowerShell CIM memory checks with a Python Windows API memory check.
- A split-grouped runner was added for future scheduled runs: `src/utils/run_baseline_grouped_batch.py`.

## Important Blockers

- Mandatory GEARS model is not complete. Current blocker:
  - missing packages: `torch_geometric`, `gears`
  - complete submission package must not be marked finished until GEARS is installed and run, or the user explicitly changes the requirement.
- Full baseline rerun is still in progress. Continue with small batches rather than `run-all`.

## Safe Continuation Commands

Recommended low-cost continuation:

```powershell
E:\anaconda3\envs\sw_mgli\python.exe src\utils\run_baseline_batch.py --manifest results\tables\baseline_run_matrix.csv --max-runs 10 --models control_mean_baseline global_delta_baseline perturbation_mean_delta_baseline --timeout-minutes 45 --cooldown-seconds 3 --kill-zombies
```

kNN continuation:

```powershell
E:\anaconda3\envs\sw_mgli\python.exe src\utils\run_baseline_batch.py --manifest results\tables\baseline_run_matrix.csv --max-runs 5 --models cell_context_knn_delta_baseline --timeout-minutes 60 --cooldown-seconds 5 --kill-zombies
```

Ridge continuation, intentionally slow and small:

```powershell
E:\anaconda3\envs\sw_mgli\python.exe src\utils\run_baseline_batch.py --manifest results\tables\baseline_run_matrix.csv --max-runs 1 --models ridge_regression_baseline --timeout-minutes 90 --cooldown-seconds 5 --kill-zombies
```

Never use one-shot `run-all` on the expanded 525-run manifest on this workstation.

For the remaining dataset-heldout block, use a dedicated long-running window after either optimizing all-dataset loading or accepting overnight execution. Do not run it during interactive work:

```powershell
E:\anaconda3\envs\sw_mgli\python.exe src\utils\run_baseline_batch.py --manifest results\tables\baseline_run_matrix.csv --max-runs 1 --max-total-runs 1 --split-families dataset_heldout_split --timeout-minutes 180 --cooldown-seconds 15 --min-free-gb 6 --kill-zombies
```

## 2026-04-27 Long-Run Continuation

- The dataset-heldout baseline path was optimized so all-dataset splits read only the required global-intersection gene columns instead of repeatedly materializing every full native gene matrix.
- `run_campaign` was fixed to respect a caller-provided temporary manifest. Before this fix, `run-all --manifest <temp>` rebuilt and overwrote the temporary manifest with the full 525-row campaign, which could silently turn a small grouped run into an unbounded long run.
- Light validation after the fix:
  - `python -m py_compile src\baselines\run_baseline.py src\utils\run_baseline_grouped_batch.py`
  - `python -m pytest tests\splits tests\models tests\data tests\evaluation\test_metrics.py tests\transportability\test_ptl.py -q`
  - Result: `16 passed`.
- A guarded grouped long-run was started with one split group at a time:

```powershell
E:\anaconda3\envs\sw_mgli\python.exe src\utils\run_baseline_grouped_batch.py --manifest results\tables\baseline_run_matrix.csv --max-groups 10 --split-families dataset_heldout_split --timeout-minutes 180 --pause-between-groups-seconds 30 --min-free-gb 6 --kill-zombies --log-file results\logs\training\phase11_baseline_batch.log
```

- DatlingerBock2021 seed0 and seed1 completed before the manifest fix was applied and the accidental full-manifest process was stopped after the seed1 ridge output was safely written.
- Current post-fix continuation starts from DatlingerBock2021 seed2 and will continue through the remaining dataset-heldout groups unless a process or memory guard stops it.
