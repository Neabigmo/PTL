# Phase 09 - Figures and Visual Design

## Objective
Create publication-quality figures with consistent scientific visual style.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/09_figures_and_visual_design.md
- results/tables/main_findings.csv
- results/tables/transfer_decay_summary.csv
- results/tables/selective_prediction_summary.csv
- results/tables/failure_mode_atlas.csv
- docs/results_narrative.md

## Output files
- results/figures/fig1_concept.html and .png
- results/figures/fig2_benchmark.html and .png
- results/figures/fig3_decay_heatmap.png
- results/figures/fig4_ptl_architecture.html and .png
- results/figures/fig5_selective_prediction.png
- results/figures/fig6_failure_modes.png
- docs/figure_notes/*.md
- results/logs/figures/phase09_figures.log
- docs/task_logs/phase09_figures_and_visual_design.md

## Detailed steps
1. Use HTML/CSS/SVG for conceptual flowcharts where possible.
2. Use restrained journal-style colors, near-white background, readable typography, and no default plotting styles.
3. Create six required figures: concept, benchmark splits, decay heatmap, PTL architecture, selective prediction curves, and failure-mode atlas.
4. Export source files plus PNG, and PDF/SVG where feasible.
5. Write one figure note per figure with message, data source, and limitations.
6. Check for overlapping labels and unreadable text.
7. Update tracking files and task log.

## Commands
Example pattern:

```powershell
python src/figures/make_figures.py
```

## Useful plugin/tool support
- scientific-visualization, scientific-schematics, matplotlib, seaborn, and plotly skills for figure generation.
- Browser Use plugin for local HTML/SVG inspection, desktop/mobile viewport checks, and visual QA.

## Logging requirements
- Save figure-generation log to results/logs/figures/phase09_figures.log.

## Acceptance criteria
- Every required figure has a source file where applicable.
- Every figure has a note.
- Figures are visually consistent and readable.
- Tracking files are updated.

## Next task dependency
Phase 10 references figures in the manuscript draft.
