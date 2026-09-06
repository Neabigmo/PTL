# Phase 01 - Literature and Positioning

## Objective
Build the scientific story, literature map, research gap, and positioning for perturbation transportability in virtual-cell and single-cell perturbation prediction.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/01_literature_and_positioning.md

## Output files
- docs/literature_map.md
- docs/research_gap.md
- docs/positioning_statement.md
- docs/task_logs/phase01_literature_and_positioning.md

## Detailed steps
1. Search current literature on virtual cell models, single-cell foundation models, perturbation prediction, Perturb-seq, GEARS, CPA, scGen, scGPT, Geneformer, calibration, abstention, selective prediction, transportability, domain generalization, and benchmark design.
2. Summarize at least 30 relevant papers or resources with one-line relevance each.
3. Identify what existing studies evaluate well and what they miss about context-dependent transfer.
4. Write a clear research gap centered on lack of model-agnostic transportability definition, stress benchmarks, and reliability calibration.
5. Write a concise positioning statement including 3 to 5 manuscript contributions.
6. Add target-journal angles for EAAI, KBS, AIME, JBI, and CMPB.
7. Update project_status.md, file_registry.md, decision_log.md, and task log.

## Commands
Use web/literature search tools as needed. Prefer scite literature MCP for peer-reviewed claims and citation verification. Save notes to files, not chat.

## Useful plugin/tool support
- scite literature MCP for paper discovery, full-text excerpts, and verified citations.
- literature-review, paper-lookup, research-lookup, and citation-management skills if they match the immediate subtask.
- Browser/search tools only for current resource discovery or access checks.
- If callable, use Life Science Research plugin skills such as Research Router, bioRxiv / medRxiv, NCBI PMC, NCBI Entrez, CELLxGENE, Open Targets, Reactome, STRING, Human Protein Atlas, GWAS Catalog, ChEMBL, and UniProt for targeted background evidence and resource summaries.

## Logging requirements
- Save literature search notes and decisions in docs/task_logs/phase01_literature_and_positioning.md.
- Do not paste long bibliographies into chat.

## Acceptance criteria
- At least 30 papers/resources summarized.
- Research gap is explicit and bounded.
- Contribution list has 3 to 5 bullets.
- Target-journal positioning is written.
- Tracking files are updated.

## Next task dependency
Phase 02 uses the positioning to prioritize datasets and benchmark framing.
