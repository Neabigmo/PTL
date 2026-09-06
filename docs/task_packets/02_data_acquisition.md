# Phase 02 - Data Acquisition

## Objective
Acquire or prepare public single-cell perturbation datasets with complete inventory and download logging.

## Required input files
- PROJECT_RULES.md
- docs/project_status.md
- docs/file_registry.md
- docs/task_packets/02_data_acquisition.md
- docs/literature_map.md if available

## Output files
- results/tables/data_inventory.csv
- docs/data_inventory.md
- results/logs/downloads/*.log
- data/raw/ or data/external/ dataset files
- docs/task_logs/phase02_data_acquisition.md

## Detailed steps
1. Check whether the preferred conda environment exists before running Python.
2. Prioritize scPerturb harmonized data, Norman combinatorial Perturb-seq, and Replogle K562/RPE1 Perturb-seq.
3. Optionally inspect Virtual Cell Challenge, CELLxGENE reference data, and public embedding resources only if feasible.
4. For each dataset, record source URL, version/date, organism, cell type/line, perturbation type, size fields, license/access notes, download method, and local path.
5. For every download attempt, write a log with timestamp, command, output path, proxy status, file size if successful, checksum if feasible, and errors if any.
6. If a download fails, retry once with proxy. If still blocked, update docs/manual_intervention_needed.md with exact URL, required file, and user action.
7. Update tracking files and task log.

## Commands
Only run download commands after confirming the environment and logging plan.

## Useful plugin/tool support
- research-lookup, parallel-web, or Browser Use may help discover official dataset pages and verify access routes.
- cellxgene-census may be useful only for optional CELLxGENE reference data.
- If callable, use Life Science Research plugin skills CELLxGENE, BioStudies / ArrayExpress, ENCODE, NCBI Datasets, and NCBI Entrez to identify official dataset pages and metadata.
- All downloads still require logs under results/logs/downloads/ and must follow proxy/manual-intervention rules.

## Logging requirements
- Every download attempt must have a log under results/logs/downloads/.
- Do not silently skip unavailable datasets.

## Acceptance criteria
- At least one Tier 1 dataset is available locally, or blockers are fully documented.
- data_inventory.csv and docs/data_inventory.md exist.
- All download attempts are logged.
- Tracking files are updated.

## Next task dependency
Phase 03 requires at least one accessible dataset or explicit user intervention.
