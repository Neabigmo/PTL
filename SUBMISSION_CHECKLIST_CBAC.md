# CBAC pre-submission checklist

## Main manuscript
- Single-anonymized title page targets Computational Biology and Chemistry.
- Main text contains only Figures 1-5; S1-S4 are in the supplementary source.
- Abstract is under 250 words and frames PTL as reliability modeling for functional genomics predictions.
- Data/code availability points to the minimal PTL repository and public data sources.

## Required submission files
- `main_text/main.tex` and compiled main manuscript PDF.
- `main_text/highlights.txt` with 3-5 highlights, each <=85 characters.
- `main_text/keywords.txt` with 1-7 keywords.
- `main_text/graphical_abstract_caption.txt`
- `supplementary/supplementary_figures.tex` and compiled supplementary PDF.
- `figures/cbac/` contains deterministic PDF/SVG/PNG/TIFF figure exports and graphical abstract.

## Evidence tables
- `additional_public_screen_validation_summary.csv`
- `external_candidate_contract_audit.csv`
- `methods_comparison_table.csv`
- `methods_submission_compliance_summary.json`
- `cbac_case_examples.csv`
- `cbac_case_exclusion.csv`
- `cbac_submission_compliance_summary.json`

## Scope wording
- No obsolete target-journal wording.
- GSE264667/NadigOConner2024 described only as an audited external candidate.
- GEARS is described as adapter/output-contract validation, not a full benchmark.
- Graphical abstract and figures are deterministic plotting/vector outputs, not generative-image outputs.

## Repository hygiene
- Minimal repository excludes raw data, large model outputs, generated figures, caches, and `.pyc` files.
- Expected outputs are documented in `github_minimal_repo/EXPECTED_OUTPUTS.md`.
