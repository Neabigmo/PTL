# Phase 07 - PTL Model

## Objective
Build PTL, a model-agnostic reliability layer that predicts whether a perturbation model output is transportable across context-stress settings.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/07_ptl_model.md
- results/tables/all_metrics.csv
- Baseline outputs and split diagnostics
- data/processed/ metadata and signatures

## Output files
- src/transportability/ptl.py
- results/tables/ptl_metrics.csv
- results/tables/ptl_ablation.csv
- results/tables/failure_mode_summary.csv
- results/logs/training/phase07_ptl.log
- docs/task_logs/phase07_ptl_model.md

## Detailed steps
1. Define PTL examples from model, split, context, perturbation, and metric records.
2. Build features for uncertainty, train-test context distance, perturbation novelty, cell-state novelty, combination novelty, DEG support, pathway support if available, replicate/support count, and dataset identity or batch proxy.
3. Define binary, regression, and optional failure-mode targets from observed prediction fidelity.
4. Train logistic regression, random forest, and XGBoost/LightGBM if available. Use small MLP only if justified.
5. Produce transportability_score, keep_or_abstain decision, and explanation table.
6. Run ablations: no_context_distance, no_perturbation_novelty, no_pathway_features, no_uncertainty_features, full_PTL.
7. Update tracking files and task log.

## Commands
Example pattern:

```powershell
python src/transportability/train_ptl.py --metrics results/tables/all_metrics.csv
```

## Useful plugin/tool support
- scikit-learn skill for logistic regression, random forest, calibration, and model selection.
- shap skill for explanation tables if the installed environment supports it.
- If callable, use Life Science Research plugin skills Reactome, STRING, Open Targets, GTEx eQTL, eQTL Catalogue, Human Protein Atlas, UniProt, and Ensembl to build optional biological-support features. Keep these features as ablations and do not require them for the core PTL if access is blocked.

## Logging requirements
- Save PTL training log to results/logs/training/phase07_ptl.log.
- Append PTL runs to experiment_registry.csv if model training occurs.

## Acceptance criteria
- PTL improves selective prediction over naive confidence.
- Ablation table exists.
- Failure mode summary exists.
- Tracking files are updated.

## Next task dependency
Phase 08 synthesizes PTL and benchmark results.
