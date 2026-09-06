# Phase 08 - Main Analysis

## Objective
Answer the manuscript's main scientific questions using benchmark, baseline, metric, and PTL outputs.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/08_main_analysis.md
- results/tables/all_metrics.csv
- results/tables/ptl_metrics.csv
- results/tables/ptl_ablation.csv
- results/tables/failure_mode_summary.csv

## Output files
- results/tables/main_findings.csv
- results/tables/transfer_decay_summary.csv
- results/tables/selective_prediction_summary.csv
- results/tables/failure_mode_atlas.csv
- docs/results_narrative.md
- results/logs/phase08_main_analysis.log
- docs/task_logs/phase08_main_analysis.md

## Detailed steps
1. Quantify performance drops from random split to context-stress splits.
2. Test whether model rankings change across split families.
3. Identify biological contexts with the largest failures.
4. Assess whether PTL predicts likely failure.
5. Evaluate selective prediction and retained-prediction fidelity.
6. Summarize dominant failure modes.
7. Write a bounded results narrative that avoids clinical, wet-lab, universal-best-model, de novo drug discovery, or final-truth claims.
8. Update tracking files and task log.

## Commands
Example pattern:

```powershell
python src/evaluation/main_analysis.py --metrics results/tables/all_metrics.csv
```

## Useful plugin/tool support
- statsmodels skill for statistical comparisons if needed.
- matplotlib, seaborn, or plotly skills for figure-ready exploratory summaries.
- If callable, use Life Science Research plugin skills Reactome, STRING, Open Targets, Human Protein Atlas, and selected genetics resources only for bounded failure-mode interpretation.

## Logging requirements
- Save analysis log to results/logs/phase08_main_analysis.log.
- Save detailed notes to docs/task_logs/, not chat.

## Acceptance criteria
- Every main question is answered by a table or figure-ready output.
- Results narrative is bounded.
- Limitations are documented.
- Tracking files are updated.

## Next task dependency
Phase 09 turns analysis outputs into figures.
