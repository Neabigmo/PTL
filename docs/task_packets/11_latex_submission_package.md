# Phase 11 - LaTeX and Submission Package

## Objective
Compile the manuscript and prepare a submission package with logs and diagnostics.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/11_latex_submission_package.md
- manuscript/main.tex
- manuscript/sections/*.tex
- manuscript/references.bib
- results/figures/
- results/tables/

## Output files
- manuscript/main.pdf
- results/logs/latex/main_compile.log
- results/logs/latex/latex_compile_diagnosis.md if errors occur
- submit/manuscript.pdf
- submit/figures/
- submit/tables/
- submit/cover_letter.md
- submit/highlights.md if needed
- submit/graphical_abstract.png if created
- submit/data_code_availability.md
- docs/task_logs/phase11_latex_submission_package.md

## Detailed steps
1. Add D:\texlive\2025\bin\windows to PATH if needed.
2. Compile from the manuscript directory using latexmk when available.
3. If latexmk is unavailable, try pdflatex with nonstop mode and halt-on-error.
4. Save full compile logs under results/logs/latex/.
5. If compilation fails, write concise diagnosis, fix only necessary source files, and recompile.
6. Prepare submit/ package with PDF, figures, tables, cover letter, and availability statement.
7. Update tracking files and task log.

## Commands
```powershell
$env:Path = "D:\texlive\2025\bin\windows;$env:Path"
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Useful plugin/tool support
- Browser Use plugin for local inspection of final HTML/SVG/PDF-adjacent artifacts when applicable.
- pdf skill for PDF review if the compiled manuscript requires structured inspection.

## Logging requirements
- Never delete .log files.
- Save errors and fixes in results/logs/latex/ and docs/task_logs/.

## Acceptance criteria
- PDF compiles successfully.
- Submission package exists.
- Compile logs exist.
- Tracking files are updated.

## Next task dependency
This is the final project phase unless revisions are requested.
