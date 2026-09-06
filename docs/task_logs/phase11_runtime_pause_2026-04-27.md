# Phase 11 Opinion 3 Runtime Pause Record

Paused: 2026-04-27 Asia/Shanghai

## User Instruction

The user explicitly stopped the current run and requested that no further computation be launched. This document records the current state of the Opinion 3 reinforcement work, especially the long-running baseline reruns and the remaining blockers.

## Safety Action Taken

After the stop request, the following project Python processes were terminated to avoid continued background resource use:

- `src/utils/run_baseline_grouped_batch.py --manifest results/tables/baseline_gene_count_mismatch_run_matrix.csv --max-groups 12 ... --force`
- `src/baselines/run_baseline.py run-all --manifest ...baseline_group_0bo1giid.csv --force`

No additional training, evaluation, plotting, or manuscript compilation was started after the stop request.

## Major Work Completed Before Pause

### Expanded Dataset Scope

The project had already been expanded from the earlier small benchmark to 9 processed datasets:

- 8 scPerturb datasets:
  - `NormanWeissman2019_filtered`
  - `ReplogleWeissman2022_K562_essential`
  - `AdamsonWeissman2016_GSM2406677_10X005`
  - `DatlingerBock2017`
  - `DatlingerBock2021`
  - `DixitRegev2016_K562_TFs_7_days`
  - `PapalexiSatija2021_eccite_RNA`
  - `ReplogleWeissman2022_rpe1`
- 1 external holdout:
  - `GSE284197_screen`

Split artifacts were rebuilt on the expanded dataset surface. DatlingerBock2021 preprocessing was repaired by changing the delta reference stratum from `sample` to `perturbation_2`, restoring non-control signatures for that dataset.

### Baseline Long-Run Completion

The transparent baseline run matrix was rebuilt to 525 planned runs. The long-running `dataset_heldout_split` block was completed using guarded grouped execution:

- Final confirmed transparent baseline status before later mismatch audit:
  - 525 / 525 baseline rows had `run_metrics.json`.
  - The grouped dataset-heldout continuation completed 50 / 50 remaining rows with `failed=0`.

This was achieved after two code-level safety fixes:

- `src/baselines/run_baseline.py` now reads only the needed gene subset for all-dataset splits instead of repeatedly materializing every full native gene matrix.
- `src/baselines/run_baseline.py run_campaign()` was fixed so a caller-provided temporary manifest is respected. Before this fix, `run-all --manifest <temp>` could overwrite the temporary manifest with the full campaign.

### Gene-Space Correction

After adding the new datasets, the global intersection gene count changed:

- Old value: `7880`
- Current expanded value: `6257`

`src/baselines/run_baseline.py` was updated so `gene_space_policy` is now dynamically named, e.g. `global_intersection_6257`, instead of the stale `global_intersection_7880`.

The rebuilt manifest confirmed:

- `dataset_heldout_split`: `global_intersection_6257`
- `external_holdout`: `global_intersection_6257`
- native within-dataset splits retain native gene counts.

## Why Additional Reruns Started

Phase 06 evaluation failed because some older all-dataset outputs had been generated before the expanded 6257-gene global intersection was adopted. The failure appeared at:

```text
dataset_heldout_split__all_datasets__holdout-GSE284197_screen__seed0__signature__control_mean_baseline
Expected gene count mismatch
```

A mismatch audit identified 60 outputs with old `gene_count=7880` where the current manifest expects `6257`:

- `dataset_heldout_split / GSE284197_screen`: 15 rows
- `dataset_heldout_split / NormanWeissman2019_filtered`: 15 rows
- `dataset_heldout_split / ReplogleWeissman2022_K562_essential`: 15 rows
- `external_holdout / GSE284197_screen`: 15 rows

A temporary rerun manifest was written:

- `results/tables/baseline_gene_count_mismatch_run_matrix.csv`

The rerun was launched with:

```powershell
E:\anaconda3\envs\sw_mgli\python.exe src\utils\run_baseline_grouped_batch.py --manifest results\tables\baseline_gene_count_mismatch_run_matrix.csv --max-groups 12 --timeout-minutes 180 --pause-between-groups-seconds 30 --min-free-gb 6 --kill-zombies --force --log-file results\logs\training\phase11_baseline_batch.log
```

## Last Confirmed Rerun State Before Stop

The mismatch rerun had successfully completed at least:

- Group 1 / 12:
  - `dataset_heldout_split__all_datasets__holdout-GSE284197_screen__seed0__signature`
  - 5 / 5 rows completed with `gene_count=6257`.
- Group 2 / 12:
  - `dataset_heldout_split__all_datasets__holdout-GSE284197_screen__seed1__signature`
  - Completed according to grouped stdout.
- Group 3 / 12:
  - `dataset_heldout_split__all_datasets__holdout-GSE284197_screen__seed2__signature`
  - Was in progress at the time of the last check.

Last confirmed mismatch audit before the stop showed:

- Remaining mismatch or missing rows: 46 / 60.
- The active run was the GSE284197 dataset-heldout seed2 ridge run, already loaded at `genes=6257` and in validation.

Because the user interrupted the following long wait and then requested no more execution, the exact final state after process termination was not rechecked. Treat all mismatch rows beyond the last confirmed state as not yet formally verified.

## Current Pipeline State

### Completed

- Expanded preprocessing and splits.
- Transparent baseline training for all 525 planned rows at least once.
- All newly added dataset-heldout rows for DatlingerBock2021, Dixit, Papalexi, and Replogle RPE1 completed safely under guarded grouped execution.
- Baseline manifest rebuilt with dynamic `global_intersection_6257`.
- Baseline summarization was run after manifest rebuild.
- Lightweight validation passed:
  - `python -m py_compile src/baselines/run_baseline.py src/utils/run_baseline_grouped_batch.py`
  - selected test subset: `16 passed`.

### Incomplete Or Not Yet Verified

- The 60 old 7880-gene outputs were only partially rerun before the stop.
- Phase 06 full evaluation did not complete after the expanded gene-space correction.
- Biological metric recomputation did not run after the corrected outputs.
- PTL training with `--feature-mode deployment --write-feature-audit` did not rerun after the expanded metrics.
- Phase 08 main analysis, Phase 09 figures, manuscript text refresh, and submission package refresh did not rerun after the expanded results.
- GEARS remains unresolved and mandatory under Opinion 3.

## GEARS Blocker

The mandatory representative perturbation model is still blocked. The adapter exists, but the environment lacks:

- `torch_geometric`
- `gears`

Existing blocker logs were written under `results/logs/training/`, and `docs/manual_intervention_needed.md` records the issue. The manuscript should not be marked as satisfying Opinion 3's model-extension requirement until GEARS is installed and run, or the requirement is explicitly changed by the user.

## Recommended Next Resume Point

When the user decides to continue, do not start with Phase 06. First verify the mismatch rerun state, then resume only the remaining rows in `results/tables/baseline_gene_count_mismatch_run_matrix.csv`.

Recommended resume order:

1. Check for project Python processes and stop stale ones if needed.
2. Recompute the mismatch audit only, without training.
3. If mismatch rows remain, rerun only those rows with grouped, forced execution.
4. Confirm all 525 baseline rows match the current manifest gene count.
5. Run `src/evaluation/evaluate_phase06.py`.
6. Run biological metrics, PTL, main analysis, figures, and manuscript/package refresh.
7. Resolve GEARS before claiming full Opinion 3 completion.

Do not use one-shot full-manifest `run-all` on this workstation.
