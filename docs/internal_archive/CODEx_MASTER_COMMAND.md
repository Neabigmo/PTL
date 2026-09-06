# CODEx_MASTER_COMMAND.md
# Master command for Codex

You are Codex working on a full AI-for-bio manuscript project.

You must first read:

1. PROJECT_RULES.md

Then execute this master command.

Important:
Do not try to complete the entire project in one context.
Your first task is to split this master plan into small task packet files.
After that, execute the project phase by phase using only the relevant task packet and required status files.

## Project title

Perturbation Transportability Lens for Virtual-Cell Models

## Working manuscript title

Context governs perturbation transfer: a calibrated benchmark for stress-testing virtual-cell models across unseen cellular states

## Scientific objective

Build a benchmark-first and reliability-first AI-for-bio paper.

The central question is:

When is a predicted single-cell perturbation response transportable across unseen biological contexts?

The work should not be framed as merely another perturbation prediction model.

It should be framed as:
1. a definition of perturbation transportability,
2. a context-stress benchmark for virtual-cell models,
3. a model-agnostic calibration layer called PTL,
4. a failure-mode atlas explaining where and why perturbation predictions fail.

## Mandatory first action

Before downloading data, installing packages, writing models, or running experiments, create the following files:

docs/task_packets/00_project_decomposition.md
docs/task_packets/01_literature_and_positioning.md
docs/task_packets/02_data_acquisition.md
docs/task_packets/03_preprocessing_and_harmonization.md
docs/task_packets/04_context_stress_splits.md
docs/task_packets/05_baselines.md
docs/task_packets/06_metrics.md
docs/task_packets/07_ptl_model.md
docs/task_packets/08_main_analysis.md
docs/task_packets/09_figures_and_visual_design.md
docs/task_packets/10_manuscript_writing.md
docs/task_packets/11_latex_submission_package.md

Also create:

docs/project_status.md
docs/file_registry.md
docs/decision_log.md
docs/manual_intervention_needed.md
docs/task_logs/
results/logs/
results/tables/
results/figures/
results/checkpoints/
data/raw/
data/interim/
data/processed/
data/external/
src/

Each task packet must be concise and executable.

Each task packet must contain:
1. objective
2. required input files
3. output files
4. detailed steps
5. commands to run if applicable
6. logging requirements
7. acceptance criteria
8. next task dependency

## Token-saving execution protocol

After creating all task packets:
1. Read only PROJECT_RULES.md, docs/project_status.md, docs/file_registry.md, and the current task packet.
2. Execute that task.
3. Save all detailed notes to files.
4. Update project_status.md.
5. Move to the next task packet.
6. Do not keep all previous task details in the active context.

## Project directory structure

Create this structure:

project/
  PROJECT_RULES.md
  CODEx_MASTER_COMMAND.md
  data/
    raw/
    interim/
    processed/
    external/
  src/
    data/
    features/
    splits/
    baselines/
    models/
    transportability/
    evaluation/
    figures/
    utils/
  results/
    tables/
    figures/
    logs/
      downloads/
      training/
      evaluation/
      latex/
      figures/
    checkpoints/
  docs/
    task_packets/
    task_logs/
    figure_notes/
    project_status.md
    file_registry.md
    decision_log.md
    manual_intervention_needed.md
    literature_map.md
    research_gap.md
    positioning_statement.md
    benchmark_definition.md
    metric_dictionary.md
    model_card.md
    manuscript_outline.md
  manuscript/
    main.tex
    sections/
    figures/
    tables/
    references.bib

## Environment initialization

Use the preferred environment:

conda activate E:\anaconda3\envs\sw_mgli

Set proxy when installing packages or downloading data.

For Windows CMD:

set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897

For PowerShell:

$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"

For LaTeX compilation, use:

D:\texlive\2025

Add this to PATH if needed:

D:\texlive\2025\bin\windows

Recommended LaTeX command:

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

## Phase 00: Project decomposition

Create all task packets listed above.

Also create:
- docs/project_status.md
- docs/file_registry.md
- docs/decision_log.md
- docs/manual_intervention_needed.md

project_status.md must include:
- current phase
- completed phases
- active blockers
- next action
- last updated timestamp

file_registry.md must include:
- file path
- purpose
- produced by which phase
- status

decision_log.md must include:
- decision
- reason
- date
- affected files

manual_intervention_needed.md must include:
- issue
- required user action
- URL or file needed
- priority
- status

Acceptance criteria:
- all task packet files exist
- all tracking files exist
- project directory structure exists
- no experiment has been started yet

## Phase 01: Literature and positioning

Goal:
Build the scientific story and gap.

Search and summarize:
1. virtual cell models
2. single-cell foundation models
3. single-cell perturbation prediction
4. Perturb-seq datasets
5. GEARS, CPA, scGen, scGPT, Geneformer
6. calibration, abstention, selective prediction
7. transportability and domain generalization in biological prediction
8. benchmark design for single-cell perturbation models

Write:
- docs/literature_map.md
- docs/research_gap.md
- docs/positioning_statement.md

The positioning statement must argue:

Current virtual-cell models are increasingly powerful, but the field lacks a systematic and model-agnostic way to define, quantify, and predict when perturbation responses are transportable across biological contexts.

Acceptance criteria:
- at least 30 relevant papers or resources summarized
- each paper/resource has one-line relevance
- research gap is written clearly
- manuscript contribution is stated in 3 to 5 bullet points
- target journal angle is written for EAAI, KBS, AIME, JBI, CMPB

## Phase 02: Data acquisition

Goal:
Acquire or prepare public single-cell perturbation datasets.

Priority datasets:
1. scPerturb harmonized datasets
2. Norman et al. combinatorial Perturb-seq
3. Replogle et al. K562/RPE1 Perturb-seq
4. optional: Virtual Cell Challenge public data if accessible
5. optional: CELLxGENE reference data for external cell-state representation
6. optional: scGPT/Geneformer embeddings if feasible

For each dataset, record:
- source
- URL
- version/date
- perturbation type
- organism
- cell type or cell line
- number of cells
- number of perturbations
- number of genes
- license/access note
- download method
- whether manual download is needed

Required outputs:
- results/tables/data_inventory.csv
- docs/data_inventory.md
- results/logs/downloads/*.log
- data/raw/ or data/external/ files

If download fails:
- retry once with proxy
- if still failing, write manual instructions to docs/manual_intervention_needed.md
- do not silently skip

Acceptance criteria:
- at least one Tier 1 dataset is available locally
- data inventory table exists
- every download attempt has a log
- blockers are documented

## Phase 03: Preprocessing and harmonization

Goal:
Convert datasets into analysis-ready files.

For each dataset:
1. load AnnData or equivalent
2. standardize gene symbols
3. identify control cells
4. identify perturbation labels
5. identify cell type/cell line/state labels
6. filter low-quality cells
7. remove extremely rare perturbations unless used only for low-support analysis
8. compute pseudobulk signatures
9. compute perturbation-effect vectors:
   delta = mean(expression_perturbed) - mean(expression_control)
10. save metadata and processed matrices

Required outputs:
- data/processed/{dataset}_adata.h5ad
- data/processed/{dataset}_perturbation_signatures.parquet
- data/processed/{dataset}_metadata.parquet
- results/tables/preprocessing_summary.csv
- results/logs/phase03_preprocessing.log

Acceptance criteria:
- at least one dataset is fully processed
- control and perturbation labels are validated
- summary table exists
- preprocessing script is reusable

## Phase 04: Context-stress benchmark splits

Goal:
Create benchmark split families that test transportability.

Implement:
1. random_split
2. unseen_perturbation_split
3. unseen_combination_split
4. unseen_cell_state_split
5. unseen_cell_type_or_lineage_split
6. dataset_heldout_split if multiple datasets are available
7. low_support_split

For every split:
- save train/val/test indices
- report number of cells
- report number of perturbations
- report number of combinations
- report number of cell states/cell types
- calculate train-test distance diagnostics
- calculate leakage diagnostics

Required outputs:
- data/processed/splits/*.json
- results/tables/split_audit.csv
- docs/benchmark_definition.md
- results/logs/phase04_splits.log

Acceptance criteria:
- every split has reproducible saved indices
- split audit table exists
- leakage diagnostics are reported
- random split and at least two stress splits are ready

## Phase 05: Baseline models

Goal:
Train stable baselines before advanced models.

Implement:
1. control_mean_baseline
2. global_delta_baseline
3. perturbation_mean_delta_baseline
4. cell_context_knn_delta_baseline
5. ridge_regression_baseline
6. LightGBM or XGBoost baseline if feature tables are available

Optional if dependencies are manageable:
1. scGen
2. CPA
3. GEARS

Do not over-engineer optional models.
The key contribution is the benchmark and PTL reliability layer.

Required outputs:
- src/baselines/
- results/tables/baseline_metrics_by_split.csv
- results/tables/model_ranking_instability.csv
- results/logs/training/*.log
- results/tables/experiment_registry.csv

Acceptance criteria:
- at least four baselines are implemented
- each baseline runs on random split and stress splits
- training logs exist
- experiment registry is updated

## Phase 06: Evaluation metrics

Goal:
Implement multi-level evaluation beyond MSE.

Metrics:

Expression fidelity:
- Pearson
- Spearman
- RMSE
- MAE

Perturbation-effect fidelity:
- delta vector correlation
- top DEG overlap
- DEG direction consistency

Distributional fidelity:
- MMD
- energy distance
- Wasserstein distance if feasible

Ranking fidelity:
- Recall@K
- NDCG@K
- MAP@K

Transportability-specific metrics:
- random_to_stress_drop
- false_transportability_rate
- selective_risk
- coverage_at_confidence
- abstention_gain

Required outputs:
- src/evaluation/metrics.py
- docs/metric_dictionary.md
- results/tables/all_metrics.csv
- results/logs/evaluation/phase06_metrics.log

Acceptance criteria:
- metrics are unit-tested or smoke-tested
- metric dictionary explains formulas and interpretation
- all baseline results are evaluated consistently

## Phase 07: PTL model

Goal:
Build PTL, the Perturbation Transportability Lens.

PTL is not the main perturbation predictor.
PTL is a model-agnostic reliability layer that predicts whether a model output is transportable.

Input features:
1. prediction uncertainty features
2. train-test context distance
3. perturbation novelty
4. cell-state novelty
5. combination novelty
6. DEG support score
7. pathway support score if available
8. replicate/support count
9. dataset identity or batch proxy

Targets:
1. binary target: whether prediction fidelity exceeds threshold
2. regression target: observed prediction fidelity
3. optional multiclass target: failure mode

Models:
1. logistic regression
2. random forest
3. XGBoost or LightGBM
4. small MLP only if needed

Outputs:
- transportability_score
- keep_or_abstain decision
- failure_mode label if possible
- explanation table

Ablations:
- no_context_distance
- no_perturbation_novelty
- no_pathway_features
- no_uncertainty_features
- full_PTL

Required outputs:
- src/transportability/ptl.py
- results/tables/ptl_metrics.csv
- results/tables/ptl_ablation.csv
- results/tables/failure_mode_summary.csv
- results/logs/training/phase07_ptl.log

Acceptance criteria:
- PTL improves selective prediction over naive confidence
- ablation table exists
- failure mode summary exists
- all outputs are logged

## Phase 08: Main analysis

Goal:
Answer the paper's main questions.

Questions:
1. How much do models drop from random split to context-stress splits?
2. Do model rankings change across split families?
3. Which biological contexts cause the largest failures?
4. Can PTL predict when a model output is likely to fail?
5. Does selective prediction improve fidelity among retained predictions?
6. What are the dominant failure modes?

Required outputs:
- results/tables/main_findings.csv
- results/tables/transfer_decay_summary.csv
- results/tables/selective_prediction_summary.csv
- results/tables/failure_mode_atlas.csv
- docs/results_narrative.md
- results/logs/phase08_main_analysis.log

Acceptance criteria:
- every main question is answered with a table or figure
- results narrative is bounded and not overclaimed
- limitations are documented

## Phase 09: Figures and visual design

Goal:
Create publication-quality figures.

Use HTML/CSS/SVG for conceptual flowcharts when possible.

Required figures:

Figure 1:
Virtual-cell prediction and context-dependent perturbation transfer.

Figure 2:
Context-stress benchmark construction and split families.

Figure 3:
Performance decay heatmap across models and stress splits.

Figure 4:
PTL architecture.

Figure 5:
Selective prediction curves.

Figure 6:
Failure-mode atlas with representative cases.

Style requirements:
- top-journal quality
- elegant, restrained, high-end scientific style
- harmonious palette
- no default ugly plotting styles
- readable text
- clear hierarchy
- white or near-white background
- save source HTML/SVG plus exported PNG/PDF/SVG

Required outputs:
- results/figures/fig1_concept.html
- results/figures/fig1_concept.png
- results/figures/fig2_benchmark.html
- results/figures/fig2_benchmark.png
- results/figures/fig3_decay_heatmap.png
- results/figures/fig4_ptl_architecture.html
- results/figures/fig4_ptl_architecture.png
- results/figures/fig5_selective_prediction.png
- results/figures/fig6_failure_modes.png
- docs/figure_notes/*.md
- results/logs/figures/phase09_figures.log

Acceptance criteria:
- every figure has a source file
- every figure has a figure note
- figures are visually consistent
- no overlapping labels
- no unreadable text
- no low-quality default graphics

## Phase 10: Manuscript writing

Goal:
Draft the manuscript.

Manuscript narrative:
1. Virtual-cell models promise in-silico perturbation prediction.
2. Random split performance does not establish biological transportability.
3. We define context-stress testing for perturbation response prediction.
4. Existing models show structured transfer decay.
5. PTL predicts when outputs are reliable.
6. Selective abstention improves biological fidelity.
7. The framework provides a reusable reliability layer for future virtual-cell models.

Required manuscript files:
- docs/manuscript_outline.md
- docs/abstract_v1.md
- manuscript/main.tex
- manuscript/sections/introduction.tex
- manuscript/sections/results.tex
- manuscript/sections/methods.tex
- manuscript/sections/discussion.tex
- manuscript/sections/limitations.tex
- manuscript/references.bib

Required cover letters:
- docs/cover_letter_EAAI.md
- docs/cover_letter_KBS.md
- docs/cover_letter_AIME.md

Claim boundary:
Do not claim:
- clinical recommendation
- wet-lab validation
- universal best model
- de novo drug discovery
- final biological truth

Acceptance criteria:
- full manuscript draft exists
- figures and tables are referenced
- limitations are explicit
- claims are bounded
- cover letters exist

## Phase 11: LaTeX and submission package

Goal:
Compile and package the manuscript.

Use:

D:\texlive\2025

Compile command:

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Required outputs:
- manuscript/main.pdf
- results/logs/latex/main_compile.log
- results/logs/latex/latex_compile_diagnosis.md if errors occur
- submit/
  - manuscript.pdf
  - figures/
  - tables/
  - cover_letter.md
  - highlights.md if needed
  - graphical_abstract.png if created
  - data_code_availability.md

Acceptance criteria:
- PDF compiles successfully
- all figures are included
- references compile
- submission package exists
- logs exist

## Final instruction

Execute this project rigorously.

Do not simplify without user approval.

If blocked:
1. log the blocker
2. write manual intervention instructions
3. continue with tasks that do not depend on the blocker
4. do not silently replace the scientific plan

Always prioritize:
- scientific rigor
- reproducibility
- benchmark clarity
- detailed logs
- high-quality figures
- bounded claims
- token efficiency