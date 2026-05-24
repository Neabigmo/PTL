# PTL Methods Minimal Code Package

This repository contains the minimal code surface for the PTL Methods manuscript:
split generation, baseline/model output contracts, GEARS adapter code, PTL
training/evaluation, deterministic figure generation, and focused tests.

Raw datasets and large generated result tables are not stored in this repository.
They should be obtained from the public accessions listed in the manuscript and
from the processed evidence archive distributed with the submission package.

## Key entry points

- `src/splits/make_splits.py`: context-stress split manifests and split audits.
- `src/baselines/run_baseline.py`: transparent baseline output contract.
- `src/models/run_perturbation_model.py`: published-model adapter contract.
- `src/transportability/train_ptl.py`: PTL training interface.
- `src/analysis/methods_submission_compliance.py`: Methods submission summary tables.
- `figures/scripts/make_all_figures.py`: deterministic manuscript figures.

## Expected checks

```bash
python -m py_compile src/analysis/methods_submission_compliance.py
python -m pytest tests/splits tests/models tests/data tests/evaluation tests/transportability
python figures/scripts/make_all_figures.py
```

The full manuscript results require the processed evidence tables and prediction
outputs described in `manifests/`.
