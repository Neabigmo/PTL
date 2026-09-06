# Phase 05 - Baseline Models

## Objective
Implement and run stable baseline perturbation predictors before any optional advanced models.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/05_baselines.md
- data/processed/ processed matrices, metadata, signatures
- data/processed/splits/*.json
- docs/benchmark_definition.md

## Output files
- src/baselines/
- results/tables/baseline_metrics_by_split.csv
- results/tables/model_ranking_instability.csv
- results/tables/experiment_registry.csv
- results/logs/training/*.log
- docs/task_logs/phase05_baselines.md

## Detailed steps
1. Check and use the preferred conda environment if available.
2. Implement command-line runners for control_mean_baseline, global_delta_baseline, perturbation_mean_delta_baseline, cell_context_knn_delta_baseline, ridge_regression_baseline, and LightGBM/XGBoost if feature tables support it.
3. Run at least four baselines on random split and feasible stress splits.
4. Save predictions or compact evaluation-ready outputs.
5. Append every run to experiment_registry.csv.
6. Compute baseline metrics already available or defer full metric expansion to Phase 06.
7. Update tracking files and task log.

## Commands
Example pattern:

```powershell
python src/baselines/run_baseline.py --model control_mean --split random_split --seed 0
```

## Useful plugin/tool support
- scikit-learn skill for ridge, kNN, calibration helpers, and robust model evaluation patterns.
- statsmodels skill only if statistical diagnostics require it.

## Logging requirements
- Every run needs a training log under results/logs/training/.
- Record command, environment, seed, dataset, split, model, status, output path, and metrics.

## Acceptance criteria
- At least four baselines are implemented.
- Baselines run on random and stress splits.
- Training logs and experiment registry rows exist.
- Tracking files are updated.

## Next task dependency
Phase 06 evaluates baseline outputs consistently.
