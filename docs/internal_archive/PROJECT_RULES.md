# PROJECT_RULES.md
# Mandatory project rules for Codex

## 0. Project identity

Project name:
Perturbation Transportability Lens for Virtual-Cell Models

Working title:
Context governs perturbation transfer: a calibrated benchmark for stress-testing virtual-cell models across unseen cellular states

Core idea:
This project is not primarily about building another perturbation prediction model. It is about defining, benchmarking, and calibrating when virtual-cell or single-cell perturbation model predictions are transportable across unseen biological contexts.

Main contribution:
1. Define perturbation transportability.
2. Build context-stress benchmark splits.
3. Evaluate baseline and existing perturbation prediction models under random vs stress splits.
4. Build PTL, a lightweight Perturbation Transportability Lens.
5. Produce failure-mode atlas and selective prediction analysis.
6. Draft a full manuscript targeting EAAI / KBS / AIME / JBI / CMPB.

## 1. Token-saving rule

You must save tokens aggressively.

Do not keep the entire project plan in context after initialization.

At the beginning of the project, split the plan into small task files under:

docs/task_packets/

Each task packet must be self-contained and concise.

Recommended maximum length per task packet:
- 800 to 1500 words
- no unnecessary background
- only current objective, input files, output files, commands, acceptance criteria

During execution:
- Only read PROJECT_RULES.md, SKILLS_INDEX.md, docs/project_status.md, docs/file_registry.md, and the current task packet.
- Do not reread all previous task packets unless necessary.
- Summarize completed work into docs/project_status.md.
- Move detailed logs into results/logs/ and docs/task_logs/.
- Never paste large tables or long raw outputs into the chat; save them as files and summarize.

## 1A. Skill gate rule

Before planning or executing any project-level task, check:

H:\2026try\4.24\SKILLS_INDEX.md

If SKILLS_INDEX.md lists a skill that may materially help the task:
1. Stop before executing the task.
2. Tell the user which skill or skills should be added to the next prompt.
3. Ask the user to start the next conversation with `/skill` or the exact skill name.
4. Do not continue the project-level task in the current turn unless the user explicitly authorizes continuing without that skill.

If no listed skill is relevant, record that the skill gate was checked in the task log and proceed normally.

Do not load unrelated skill documentation. Use only the matched skill or router.

## 2. Environment rule

Preferred conda environment:

conda activate E:\anaconda3\envs\sw_mgli

Before running Python code, check whether this environment exists.

If it exists, use it.

If it does not exist, report this clearly and create an environment setup proposal, but do not silently switch to an unknown environment unless necessary.

All Python execution should be reproducible from command line scripts.

Use project-local scripts and avoid notebook-only workflows unless explicitly requested.

## 3. LaTeX compilation rule

Use the following TeX Live installation for LaTeX compilation:

D:\texlive\2025

Before compiling manuscript files, add the TeX Live binary path to PATH if needed:

D:\texlive\2025\bin\windows

Recommended compilation command:

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

If latexmk is unavailable, try:

pdflatex -interaction=nonstopmode -halt-on-error main.tex

Run compilation from the manuscript directory.

Save all LaTeX logs under:

results/logs/latex/

Do not delete .log files after compilation.

If compilation fails:
1. Save the full error log.
2. Write a concise diagnosis to results/logs/latex/latex_compile_diagnosis.md.
3. Fix only the necessary source files.
4. Recompile.
5. Record the fix in docs/task_logs/.

## 4. Proxy and network rule

When installing packages or downloading data, prefer using the local proxy on port 7897.

Recommended environment variables on Windows CMD:

set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897

Recommended environment variables on PowerShell:

$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"

If academic websites, journal websites, GEO/SRA, Zenodo, Figshare, CELLxGENE, or other data repositories cannot be accessed:
1. Do not abandon the task.
2. Do not replace the dataset with an unrelated one without permission.
3. Write the exact URL, error message, and required file name into:
   docs/manual_intervention_needed.md
4. Ask the user to manually download the file, configure a token, or enable campus network access.

If an API token is needed:
1. Identify the required service.
2. Explain where the token should be placed.
3. Do not hard-code tokens into source code.
4. Use .env or config/local_config.yaml, which must be ignored by git.

## 5. Data download rule

Every data download attempt must create logs.

Required log location:

results/logs/downloads/

Each download log must include:
- timestamp
- dataset name
- source URL
- command used
- output path
- file size if successful
- checksum if feasible
- error message if failed
- whether proxy was used
- whether user intervention is needed

If download fails:
- Try one safe retry with proxy enabled.
- If still failing, write a manual download instruction file.
- Do not silently skip the dataset.
- Do not use a smaller substitute dataset unless the user approves.

## 6. Training and experiment logging rule

Every model training or evaluation run must be logged.

Required log location:

results/logs/training/
results/logs/evaluation/

Every training log must include:
- timestamp
- git commit or file hash if available
- command line
- conda environment
- Python version
- package versions when feasible
- dataset version
- split name
- random seed
- model name
- hyperparameters
- metrics
- output checkpoint path
- runtime
- failure traceback if failed

Every experiment must also append one row to:

results/tables/experiment_registry.csv

Required columns:
- run_id
- timestamp
- phase
- dataset
- split
- model
- seed
- command
- status
- main_metric
- output_dir
- log_file
- notes

## 7. No simplification without approval

High-quality completion is mandatory.

Do not simplify tasks without explicit user approval.

Forbidden without approval:
- replacing context-stress splits with only random split
- dropping key metrics
- dropping external validation
- dropping logging
- dropping figure quality requirements
- replacing benchmark with only a toy dataset
- using only one baseline if more are feasible
- skipping LaTeX compilation
- skipping manuscript drafting
- ignoring failed downloads
- silently changing the scientific question

Allowed without approval:
- using a smaller smoke-test subset before full run
- writing robust fallback code
- adding diagnostics
- caching intermediate data
- adding unit tests
- making visual style more polished
- using a baseline first before heavier models

If a task is impossible due to access, compute, or package conflicts:
1. Explain the blocker.
2. Save evidence in logs.
3. Propose options.
4. Ask user for approval before simplifying.

## 8. Figure and flowchart rule

Flowcharts and conceptual diagrams should be drawn using HTML/CSS/SVG whenever possible.

Required style:
- top-journal aesthetic
- clean white or near-white background
- elegant flat-vector scientific design
- harmonious colors
- restrained palette
- high readability
- no clutter
- no ugly default colors
- consistent typography
- exportable to PNG/PDF/SVG

Recommended palette:
- muted navy: #263B5E
- desaturated blue: #4A78A8
- soft teal: #5BA99B
- muted orange: #D99A5B
- warm gray: #EAE7E1
- deep gray: #333333
- pale background: #FAFAF7

Every figure must be saved to:

results/figures/

For each figure, save:
- source HTML/SVG if applicable
- PNG
- PDF or SVG when feasible

Each figure must have a short figure note in:

docs/figure_notes/

## 9. Manuscript rule

The manuscript must maintain bounded claims.

Do not claim:
- clinical treatment recommendation
- wet-lab validation
- universal best model
- de novo drug discovery
- final biological truth

Allowed claim:
This work defines and evaluates perturbation transportability, builds a context-stress benchmark, and provides a calibrated reliability layer for virtual-cell perturbation predictions.

Manuscript outputs:
- docs/manuscript_outline.md
- manuscript/main.tex
- manuscript/sections/introduction.tex
- manuscript/sections/results.tex
- manuscript/sections/methods.tex
- manuscript/sections/discussion.tex
- manuscript/sections/limitations.tex
- manuscript/references.bib
- docs/cover_letter_EAAI.md
- docs/cover_letter_AIME.md

## 10. Project status rule

Maintain:

docs/project_status.md
docs/file_registry.md
docs/decision_log.md
docs/manual_intervention_needed.md

After every phase:
1. Update project_status.md.
2. Update file_registry.md.
3. Add decisions to decision_log.md.
4. Add unresolved issues to manual_intervention_needed.md.
5. Write a task log under docs/task_logs/.

## 11. Quality control rule

Before marking any phase complete, check:
- Are outputs saved in the expected location?
- Are logs complete?
- Are file paths recorded?
- Are errors documented?
- Are metrics reproducible?
- Are claims bounded?
- Are figures publication-quality?
- Is the next task packet clear?

A phase is not complete until its acceptance criteria are satisfied.
