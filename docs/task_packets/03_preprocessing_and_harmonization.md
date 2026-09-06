# Phase 03 - Preprocessing and Harmonization

## Objective
Convert acquired datasets into analysis-ready matrices, metadata, and perturbation signatures.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/03_preprocessing_and_harmonization.md
- results/tables/data_inventory.csv
- data/raw/ or data/external/ dataset files

## Output files
- data/processed/{dataset}_adata.h5ad
- data/processed/{dataset}_perturbation_signatures.parquet
- data/processed/{dataset}_metadata.parquet
- results/tables/preprocessing_summary.csv
- results/logs/phase03_preprocessing.log
- Reusable scripts under src/data/ and src/features/
- docs/task_logs/phase03_preprocessing_and_harmonization.md

## Detailed steps
1. Check and use the preferred conda environment if available.
2. Implement reusable command-line preprocessing scripts.
3. Load AnnData or equivalent source files.
4. Standardize gene symbols and validate expression matrix orientation.
5. Identify controls, perturbation labels, cell type/line/state labels, and batch fields where available.
6. Apply transparent quality filters and document removed cells or perturbations.
7. Compute pseudobulk summaries and perturbation-effect vectors: delta = mean(perturbed) - mean(control).
8. Save processed outputs and a preprocessing summary table.
9. Update tracking files and task log.

## Commands
Example pattern:

```powershell
python src/data/preprocess_dataset.py --config config/dataset.yaml
```

## Useful plugin/tool support
- anndata and scanpy skills for AnnData conventions, preprocessing checks, and single-cell metadata handling.
- gget or biopython skills for gene identifier support if symbol harmonization requires it.
- If callable, use Life Science Research plugin skills Ensembl, UniProt, QuickGO, EFO Ontology, Bgee, Human Protein Atlas, and NCBI Clinical Tables for gene, ontology, and expression annotation checks.

## Logging requirements
- Save full preprocessing logs to results/logs/phase03_preprocessing.log.
- Record assumptions and label mappings in task log.

## Acceptance criteria
- At least one dataset is fully processed.
- Control and perturbation labels are validated.
- Summary table exists.
- Scripts are reusable from the command line.

## Next task dependency
Phase 04 requires processed metadata, matrices, and signatures.
