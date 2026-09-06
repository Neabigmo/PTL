# Phase 04 - Context-Stress Splits

## Objective
Create reproducible benchmark split families that stress perturbation transportability across unseen biological contexts.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/04_context_stress_splits.md
- data/processed/{dataset}_metadata.parquet
- data/processed/{dataset}_perturbation_signatures.parquet

## Output files
- data/processed/splits/*.json
- results/tables/split_audit.csv
- docs/benchmark_definition.md
- results/logs/phase04_splits.log
- Scripts under src/splits/
- docs/task_logs/phase04_context_stress_splits.md

## Detailed steps
1. Implement reproducible split generation with fixed seeds.
2. Create random_split, unseen_perturbation_split, unseen_combination_split, unseen_cell_state_split, unseen_cell_type_or_lineage_split, dataset_heldout_split if multiple datasets exist, and low_support_split.
3. Save train/validation/test indices for every split.
4. Report numbers of cells, perturbations, combinations, cell states, and cell types per split.
5. Compute train-test distance diagnostics and leakage diagnostics.
6. Write docs/benchmark_definition.md with split definitions and intended stress-test interpretation.
7. Update tracking files and task log.

## Commands
Example pattern:

```powershell
python src/splits/make_splits.py --dataset DATASET --seed 0
```

## Logging requirements
- Save split generation logs to results/logs/phase04_splits.log.
- Record any unavailable split family and reason.

## Acceptance criteria
- Saved indices exist for every feasible split.
- split_audit.csv exists.
- Leakage diagnostics are reported.
- Random split and at least two stress splits are ready.

## Next task dependency
Phase 05 trains baselines against these splits.
