# PTL: reliability modeling for single-cell perturbation prediction

This repository contains the core code for the PTL manuscript:
split generation, transparent baseline/output contracts, GEARS adapter code,
PTL training/evaluation, case-example analysis, deterministic figure generation,
and focused tests.

Raw datasets are not stored in this repository. Several source datasets are
large public Perturb-seq/scPerturb/GEO resources with their own access,
license, and redistribution terms. Users must obtain those datasets directly
from the original repositories or controlled-access sources listed in
`DATA_SOURCES.md`. This repository does not redistribute raw `.h5ad`, count
matrices, cell-level metadata, or large model-output arrays.

Lightweight manifests are included to document the analysis surface and expected
outputs. Processed evidence tables for manuscript review are distributed
separately as a Zenodo/Figshare-ready archive.

## Key entry points

- `src/splits/make_splits.py`: context-stress split manifests and split audits.
- `src/baselines/run_baseline.py`: transparent baseline output contract.
- `src/models/run_perturbation_model.py`: published-model adapter contract.
- `src/transportability/train_ptl.py`: PTL training interface.
- `src/analysis/cbac_case_examples.py`: CBAC case-example extraction.
- `src/analysis/methods_submission_compliance.py`: public-screen and external-candidate summaries.
- `figures/scripts/make_cbac_figures.py`: deterministic CBAC manuscript figures.
- `figures/scripts/make_graphical_abstract_case_plots.py`: real-data scatter panels used in the graphical abstract.
- `DATA_SOURCES.md`: source datasets and redistribution policy.

## Expected checks

```bash
python -m py_compile src/analysis/methods_submission_compliance.py
python -m pytest tests/splits tests/models tests/data tests/evaluation tests/transportability
python figures/scripts/make_cbac_figures.py
python figures/scripts/make_graphical_abstract_case_plots.py
```

The full manuscript results require the processed evidence tables and prediction
outputs described in `manifests/`.

Some tests construct tiny synthetic fixtures. Tests or scripts that inspect
source `.h5ad` files require the user to download the corresponding public data
locally; raw data are intentionally excluded from the repository.
