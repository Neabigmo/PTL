# Phase 01 Task Log: Literature and Positioning

Date: 2026-04-24 21:38 Asia/Shanghai

## Inputs Read

- PROJECT_RULES.md
- SKILLS_INDEX.md
- docs/project_status.md
- docs/file_registry.md
- docs/decision_log.md
- docs/task_packets/01_literature_and_positioning.md

## Skill Gate

- SKILLS_INDEX.md was checked before execution.
- Matched tools/skills: literature-review and Scite.
- User explicitly requested implementation of the Phase 01 plan and confirmed Zotero MCP may be used if available.
- literature-review workflow was followed for scoping, thematic synthesis, citation-backed evidence mapping, and gap identification.
- Scite literature MCP was used as the primary citation verification and discovery tool.
- Zotero MCP check: tool discovery suggested possible Zotero-related availability, but no callable Zotero methods were exposed in this session. Zotero was therefore not used and this did not block the phase.

## Scite Searches Performed

1. `("single-cell" AND perturbation AND (GEARS OR CPA OR scGen OR Perturb-seq))`
2. `("single-cell foundation model" OR scGPT OR Geneformer OR scFoundation OR "single-cell transformer")`
3. `("Perturb-seq" OR "CROP-seq" OR "sci-Plex" OR "single-cell CRISPR screen")`
4. `("domain generalization" OR transportability OR "external validity" OR "out-of-distribution") AND (biology OR biomedical OR "single-cell")`
5. `("selective prediction" OR abstention OR calibration OR "uncertainty quantification" OR "conformal prediction") AND (deep learning OR biomedical OR "single-cell")`
6. Exact-title lookup for GEARS, CellOT, Geneformer, scGPT, scFoundation, CPA, and SAMS-VAE.
7. `("CPA" AND "single-cell" AND perturbation) OR ("compositional perturbation autoencoder") OR ("chemCPA") OR ("scGen" AND "perturbation responses")`
8. `("virtual cell" OR "virtual cells" OR "AI virtual cell" OR "virtual-cell") AND (model OR foundation OR simulation OR biology)`
9. `("benchmark" AND "single-cell" AND perturbation AND prediction) OR "scPerturb" OR "perturbation response prediction benchmark"`
10. Exact-title lookup for calibration, selective classification, OOD detection, and conformal prediction references.

## Outputs Produced

- docs/literature_map.md
- docs/research_gap.md
- docs/positioning_statement.md
- docs/task_logs/phase01_literature_and_positioning.md

## Summary of Findings

- The literature strongly supports a fast-growing AI virtual-cell and single-cell foundation-model landscape.
- Perturb-seq, CROP-seq, sci-Plex, targeted Perturb-seq, genome-scale Perturb-seq, and scPerturb provide enough public data structure to build stress benchmarks.
- Existing perturbation prediction models include latent arithmetic, covariate-factorized generative models, graph-based predictors, optimal transport, and causal effect estimators.
- Recent benchmarks increasingly question whether deep models outperform simple baselines and call for better standardized evaluation.
- The missing contribution is a model-agnostic transportability framework: explicit context-stress splits, transportability targets, calibrated keep-or-abstain decisions, and a cross-model failure-mode atlas.

## Acceptance Criteria Check

- At least 30 papers/resources summarized: yes, 44 entries in docs/literature_map.md.
- Each paper/resource has one-line relevance: yes.
- Research gap is explicit and bounded: yes, in docs/research_gap.md.
- Contribution list has 3 to 5 bullets: yes, 5 contribution bullets in docs/positioning_statement.md.
- Target journal angle written for EAAI, KBS, AIME, JBI, CMPB: yes.
- Tracking files updated: yes, project_status.md, file_registry.md, and decision_log.md.

## Blockers

- None blocking Phase 01.
- Zotero MCP was not callable in this session; no manual intervention required because Scite provided sufficient verified metadata.

## Next Action

Begin Phase 02: Data acquisition, using the Phase 01 positioning to prioritize scPerturb, Norman et al., Replogle et al., and related public perturbation-response resources.
