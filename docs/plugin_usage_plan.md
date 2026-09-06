# Plugin Usage Plan

Last synchronized: 2026-09-05 Asia/Shanghai

## Scope
This file records project-relevant plugins, skills, and MCP tools available in the current Codex session. It is intentionally concise for token-saving execution.

The active manuscript line is the CBAC single-anonymized submission. Historical
phase references below are retained as execution provenance; they do not change
the current data, result, or submission contract.

## Detected plugin usable for this project
1. Browser Use

## User-reported Life Science Research plugin capabilities
The user reports that a Life Science Research plugin is installed with the skills below. They are project-relevant, but this Codex session has not exposed them as direct callable tool names yet. At the start of any phase that needs them, first check whether the specific skill is callable; if not, use the equivalent local skill, MCP tool, web source, or documented API route while preserving logging rules.

| Skill group | Skills | Project phase(s) | Planned use |
|---|---|---|---|
| Literature and preprints | bioRxiv / medRxiv, NCBI PMC, NCBI Entrez, Research Router | 01, 10 | Literature discovery, preprint tracking, PMC open-access checks, broad routing. |
| Single-cell and functional genomics | CELLxGENE, BioStudies / ArrayExpress, ENCODE, GTEx eQTL, eQTL Catalogue, Bgee, Human Protein Atlas | 01, 02, 03, 07 | Dataset discovery, expression context annotation, cell-state/context features, external biological support. |
| Gene, protein, and ontology resources | Ensembl, UniProt, AlphaFold, RCSB PDB, QuickGO, EFO Ontology, RNAcentral, IPD, NCBI Datasets, NCBI Clinical Tables | 01, 03, 07, 09 | Gene/protein metadata, ontology normalization, pathway/GO annotations, figure annotations. |
| Pathways and networks | Reactome, STRING, Open Targets, EpiGraphDB, Locus-to-Gene Mapper, Rhea | 01, 07, 08, 09 | Pathway support features, network context, mechanistic interpretation, failure-mode annotation. |
| Variation and genetics | ClinVar / NCBI Variation, gnomAD, EVA, GWAS Catalog, FinnGen PheWAS, BioBank Japan PheWAS, UKB-TOPMed PheWAS, TPMI PheWAS, Genebass Gene Burden | 01, 07, 08 | Optional genetic evidence and disease-context annotation; avoid overclaiming clinical relevance. |
| Chemical, target, and pharmacology resources | ChEMBL, ChEBI, PubChem PUG, BindingDB, PharmGKB, HMDB, CIViC | 01, 07, 08 | Optional perturbation/target metadata, ligand-target annotation, pharmacogenomic context where relevant. |
| Studies, trials, proteomics, microbiome | ClinicalTrials.gov, PRIDE, ProteomeXchange, MetaboLights, MGnify | 01, 08, 10 | Background evidence only when directly relevant; not core benchmark input unless approved. |
| Sequence search | NCBI BLAST | 03 | Only if sequence identity checks become necessary; not expected for core Phase 03. |

## Useful non-plugin tools and skills
These are not all "plugins" in the strict app sense, but they can support the project when a matching phase begins.

| Capability | Planned phase(s) | Use |
|---|---|---|
| scite literature MCP | 01 | Verify peer-reviewed claims, read targeted paper excerpts, avoid citation hallucination. |
| literature-review, paper-lookup, research-lookup, citation-management skills | 01, 10 | Literature mapping, citation organization, manuscript references. |
| parallel-web / perplexity-search / Browser Use | 01, 02 | Current web/resource discovery and access checks when required. |
| cellxgene-census, anndata, scanpy skills | 02, 03 | Single-cell data discovery, AnnData handling, preprocessing workflow design. |
| gget, biopython skills | 03 | Gene symbol and biological identifier support if needed. |
| scikit-learn, statsmodels, shap skills | 05, 06, 07, 08 | Baselines, calibration, statistical summaries, interpretability. |
| matplotlib, seaborn, plotly, scientific-visualization, scientific-schematics skills | 08, 09 | Analysis plots, publication figures, conceptual diagrams. |
| Browser Use | 09, 11 | Local browser inspection of HTML/SVG figures and final artifacts. |
| scientific-writing, venue-templates, pdf skills | 10, 11 | Manuscript drafting support, venue formatting checks, PDF inspection. |

## Planned plugin usage by phase
1. Phase 09 - Figures and visual design
- Use Browser Use to open locally generated HTML/SVG figure sources and verify:
  - layout correctness
  - label overlap
  - readability on desktop and mobile viewport
  - export preview consistency

2. Phase 11 - LaTeX and submission package
- Use Browser Use to inspect final deliverables in `submit/`:
  - figure visibility and basic rendering
  - file completeness checks for submission bundle

## Operational notes
- Browser Use is for visual QA and local artifact inspection, not for data download or model training.
- Scientific claims must still be backed by retrieved literature, preferably using scite or other paper-search tools during Phase 01 and Phase 10.
- Data acquisition tools must obey PROJECT_RULES.md download logging and manual-intervention rules.
- Skills are invoked only in the phase where they are relevant; do not keep their instructions in active context across phases.
- Life Science Research skills should be used mainly for metadata, annotation, and evidence support. They should not change the core project question or replace the required single-cell perturbation benchmark.
- Keep logs of visual QA actions in:
  - `results/logs/figures/phase09_figures.log`
  - `docs/task_logs/phase09_figures_and_visual_design.md`
  - `docs/task_logs/phase11_latex_submission_package.md`

## Non-goals in this plan
- No plugin-driven data acquisition in Phase 00.
- No plugin-driven model training in any phase.
