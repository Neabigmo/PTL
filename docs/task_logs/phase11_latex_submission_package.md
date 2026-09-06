# Phase 11 LaTeX and Submission Package Log

Date: 2026-04-26 Asia/Shanghai

## Inputs checked

- `PROJECT_RULES.md`
- `SKILLS_INDEX.md`
- `docs/project_status.md`
- `docs/file_registry.md`
- `docs/task_packets/11_latex_submission_package.md`
- `manuscript/main.tex`
- `manuscript/sections/*.tex`
- `manuscript/references.bib`
- `results/figures/`
- `results/tables/`

## Skills and constraints used

- LaTeX Tectonic skill was checked for TeX compilation fallback guidance.
- PDF skill was used for PDF render and visual QA workflow.
- Venue templates skill was used for EAAI-style submission package expectations.
- Project rule preferred TeX Live path `D:\texlive\2025\bin\windows`; this was used for compilation.

## Author and venue metadata

- Target venue: Engineering Applications of Artificial Intelligence.
- Review model: double-anonymized review.
- Author: Yiheng Yang.
- Affiliations:
  - School of Future Technology, Dalian University of Technology.
  - School of Bioengineering, Dalian University of Technology.
- Corresponding email: `yangyiheng@mail.dlut.edu.cn`.
- Funding: none.
- Competing interests: none.
- Data statement: public data resources; complete raw data can be obtained from relevant platforms using the same access procedures.

## Files created or updated

- `manuscript/main.tex`
- `manuscript/title_page.tex`
- `manuscript/main.pdf`
- `manuscript/title_page.pdf`
- `results/logs/latex/main_compile.log`
- `results/logs/latex/title_page_compile.log`
- `results/logs/latex/phase11_pdf_qa.json`
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

## Compilation

Main manuscript command:

```powershell
$env:Path = "D:\texlive\2025\bin\windows;$env:Path"
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Title page command:

```powershell
$env:Path = "D:\texlive\2025\bin\windows;$env:Path"
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error title_page.tex
```

Compilation succeeded. No `latex_compile_diagnosis.md` was required because there was no fatal compile error.

## QA checks

- `manuscript/main.pdf` exists and is non-empty.
- `submit/manuscript.pdf` exists and is non-empty.
- `submit/title_page.pdf` exists and is non-empty.
- Compile logs exist under `results/logs/latex/`.
- `pdftoppm` rendered 10 anonymized manuscript pages and 1 title-page page.
- Rendered PNG files were nonblank.
- Visual spot checks were performed on the anonymized first page, figure-heavy pages, the title page, and the graphical abstract.
- The anonymized manuscript text scan found no exact author identity phrases:
  - `yiheng yang`
  - `yiheng`
  - `dalian university of technology`
  - `yangyiheng@mail.dlut.edu.cn`
  - `school of future technology`
  - `school of bioengineering`
- A standalone `Yang` hit was inspected and belongs to a cited author name in the references, not to the manuscript author identity.
- Claim scan found no forbidden project claims.
- `submit/highlights.md` contains 4 editable highlights.
- `submit/graphical_abstract.png` is 1328 x 531 px.

## Caveat

The graphical abstract is a deterministic draft assembled from local Phase 09 figure assets. It should be manually reviewed before final submission because EAAI/Elsevier has explicit restrictions on generative AI or AI-assisted artwork.

## Post-review figure adjustment 1

Applied the recommendations in `投稿前调整意见1.txt` as a post-Phase-11 manuscript refinement. The main change was to replace the earlier low-density workflow figures with a denser six-figure evidence sequence:

- Figure 1: context-stress benchmark and PTL reliability workflow.
- Figure 2: benchmark composition, split construction, and audit diagnostics.
- Figure 3: model ranking instability with heatmap, rank shift, transfer decay, external-holdout zoom, and FDR contrasts.
- Figure 4: PTL reliability filtering with architecture, ROC/AP summaries, false-transportability rates, risk--coverage behavior, ablations, and conservative risk view.
- Figure 5: failure-mode atlas with taxonomy, severity heatmap, severity composition, context dot plot, and dominant atlas rows.
- Figure 6: run matrix, seed stability, dataset/external transfer contrast, and evidence-tier audit.

Commands rerun:

```powershell
python src/figures/make_figures.py
python -m pytest tests/figures/test_make_figures.py -q
$env:Path = "D:\texlive\2025\bin\windows;$env:Path"
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Verification:

- Figure smoke tests passed: 5 passed.
- Combined tests passed in the project environment: `19 passed` for `tests/evaluation`, `tests/transportability`, and `tests/figures`.
- Refined figure PNG QA passed in `results/logs/figures/phase09_figures.log`.
- Main manuscript recompiled successfully; `manuscript/main.log` reports 11 pages and no fatal LaTeX errors.
- `submit/manuscript.pdf` was refreshed from `manuscript/main.pdf`.
- `submit/figures/` was refreshed to contain the refined six-PNG figure set plus available vector exports.
- `results/logs/latex/post_adjustment1_qa.json` records page count, identity scan, forbidden-claim scan, refined figure counts, and PNG dimensions.

## Post-review figure adjustment 2

Applied the recommendations in `投稿前调整意见2.txt`. The figure pipeline was rebuilt as a Python/matplotlib-first system, and the manuscript narrative was reduced to five main figures plus one supplementary robustness/audit figure.

Main figure sequence:

- Figure 1: study design, context-stress framing, signature benchmark pipeline, model evaluation matrix, and PTL reliability layer.
- Figure 2: benchmark composition, test-signature scale, support distribution, train--test context distance, and split audit matrix.
- Figure 3: model-by-split fidelity heatmap, rank-shift bump chart, random-to-stress decay, and dataset/external transfer zoom.
- Figure 4: PTL architecture, per-example ROC curve, precision--recall curve, false-transportability filtering, and risk--coverage behavior.
- Figure 5: failure taxonomy, model--split failure severity, severity composition, context enrichment, and representative failure cases.
- Supplementary Figure 1: run completion, seed stability, transfer contrast, and evidence-tier audit.

Key files updated:

- `src/transportability/ptl.py`
- `src/transportability/train_ptl.py`
- `src/figures/make_figures.py`
- `tests/transportability/test_ptl.py`
- `tests/figures/test_make_figures.py`
- `manuscript/sections/results.tex`
- `submit/package_manifest.md`

New/updated outputs:

- `results/tables/ptl_oof_predictions.csv`
- `results/figures/fig1_study_design.{png,pdf,svg}`
- `results/figures/fig2_benchmark_composition_audit.{png,pdf,svg}`
- `results/figures/fig3_ranking_instability_transfer_decay.{png,pdf,svg}`
- `results/figures/fig4_ptl_selective_filtering.{png,pdf,svg}`
- `results/figures/fig5_failure_mode_atlas.{png,pdf,svg}`
- `results/figures/supp_fig1_robustness_audit.{png,pdf,svg}`
- `submit/figures/` now contains only the five main figures in PNG/PDF/SVG form.
- `submit/supplementary/figures/` contains the supplementary robustness/audit figure in PNG/PDF/SVG form.

Commands rerun:

```powershell
E:\anaconda3\envs\sw_mgli\python.exe src\transportability\train_ptl.py
E:\anaconda3\envs\sw_mgli\python.exe src\evaluation\main_analysis.py --metrics results\tables\all_metrics.csv
E:\anaconda3\envs\sw_mgli\python.exe src\figures\make_figures.py
E:\anaconda3\envs\sw_mgli\python.exe -m pytest tests\evaluation tests\transportability tests\figures -q
$env:Path = "D:\texlive\2025\bin\windows;$env:Path"
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Verification:

- Combined tests passed: `19 passed`.
- Figure smoke tests after warning cleanup passed: `5 passed`.
- Figure generation completed without layout or legend warnings.
- PNG dimension/nonblank checks passed for all five main figures and the supplementary figure.
- Main manuscript recompiled successfully; `main_compile_after_adjustment2.log` reports 11 pages and no fatal LaTeX errors.
- `submit/manuscript.pdf` was refreshed from `manuscript/main.pdf`.
- PDF text extraction found no exact author identity phrases in the anonymized manuscript.
- Claim scan found no forbidden project claims.
- `pdftoppm` rendered manuscript pages 1, 5, 7, 8, and 9; all rendered PNGs were nonblank.
- `results/logs/latex/post_adjustment2_qa.json` records the adjustment 2 QA summary.

Notes:

- The large `results/tables/ptl_oof_predictions.csv` is retained locally as the source for Figure 4 curves and is not copied into the compact `submit/tables/` bundle.
- Older Phase 09 and adjustment 1 figure files remain in `results/figures/` for traceability, but the current submission package uses the five-main-figure sequence.

## 2026-04-26 Figure Spacing Adjustment

Reason:

- The five main figures were visually too compressed after the Python/matplotlib-first rebuild; several panels placed text, legends, and plotted marks too close together.

Changes:

- Increased the manuscript figure width usage from `0.95\linewidth` to `\linewidth`.
- Widened and rebalanced the main matplotlib layouts in `src/figures/make_figures.py`.
- Expanded Fig. 2 and Fig. 3 into broader rows for dense benchmark/ranking panels.
- Reallocated Fig. 4 space from the architecture schematic toward ROC/PR and selective prediction panels.
- Reworked Fig. 5 spacing by removing overlapping taxonomy subtitles, moving the severity legend away from the x-axis, and enlarging representative failure-case panels.
- Refreshed `results/figures/`, `submit/figures/`, `submit/supplementary/figures/`, and `submit/manuscript.pdf`.

Commands rerun:

```powershell
E:\anaconda3\envs\sw_mgli\python.exe src\figures\make_figures.py
E:\anaconda3\envs\sw_mgli\python.exe -m pytest tests\evaluation tests\transportability tests\figures -q
$env:Path = "D:\texlive\2025\bin\windows;$env:Path"
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Verification:

- Combined tests passed: `19 passed`.
- Figure smoke tests passed: `5 passed`.
- `manuscript/main.pdf` and `submit/manuscript.pdf` were refreshed successfully; both are 3,311,474 bytes.
- Figure pages 9-12 were rendered with `pdftoppm`; rendered PNG files were non-empty.
- Spot visual QA of Fig. 4 and Fig. 5 confirmed more open panel spacing and no missing plots.
