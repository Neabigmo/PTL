# Phase 06 - Evaluation Metrics

## Objective
Implement multi-level evaluation metrics for expression fidelity, perturbation-effect fidelity, distributional fidelity, ranking fidelity, and transportability-specific analysis.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/06_metrics.md
- Baseline predictions or evaluation-ready outputs
- data/processed/ processed truth files
- data/processed/splits/*.json

## Output files
- src/evaluation/metrics.py
- docs/metric_dictionary.md
- results/tables/all_metrics.csv
- results/logs/evaluation/phase06_metrics.log
- docs/task_logs/phase06_metrics.md

## Detailed steps
1. Implement Pearson, Spearman, RMSE, MAE, delta vector correlation, top DEG overlap, DEG direction consistency, MMD, energy distance, Wasserstein if feasible, Recall@K, NDCG@K, MAP@K, random_to_stress_drop, false_transportability_rate, selective_risk, coverage_at_confidence, and abstention_gain.
2. Add smoke tests or unit tests for metric shapes, missing values, and edge cases.
3. Evaluate all baseline outputs using one consistent interface.
4. Write metric_dictionary.md with formulas, interpretation, and failure cautions.
5. Save all_metrics.csv and logs.
6. Update tracking files and task log.

## Commands
Example pattern:

```powershell
python src/evaluation/evaluate_predictions.py --registry results/tables/experiment_registry.csv
```

## Useful plugin/tool support
- scikit-learn and statsmodels skills for metric validation, ranking metrics, and statistical summaries.
- scipy-based implementation is acceptable when already available in the project environment.

## Logging requirements
- Save evaluation log to results/logs/evaluation/phase06_metrics.log.
- Record skipped metrics and reasons.

## Acceptance criteria
- Metrics are smoke-tested or unit-tested.
- Metric dictionary is clear.
- Baseline outputs are evaluated consistently.
- Tracking files are updated.

## Next task dependency
Phase 07 uses metrics to define PTL targets and reliability labels.
