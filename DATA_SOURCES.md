# Data Sources and Redistribution Policy

This repository provides code and lightweight manifests only. It does **not**
redistribute raw public datasets, cell-level matrices, raw `.h5ad` files, or
large model-output arrays.

The manuscript analyzes public single-cell perturbation resources. Users who
want to reproduce the full analysis must download or request the source data
from the original providers and comply with the corresponding licenses, terms of
use, and access rules. This avoids unauthorized redistribution of datasets that
the author also obtained from public or third-party repositories.

## Primary public resources

| Resource | Role in manuscript | How to obtain |
| --- | --- | --- |
| scPerturb harmonized perturbation datasets | Internal public benchmark surface and additional public-screen holdouts | Obtain from the official scPerturb release and follow the provider's license/terms |
| NormanWeissman2019 | Perturb-seq / genetic-interaction benchmark component | Obtain through scPerturb or original publication data links |
| ReplogleWeissman2022 K562 / RPE1 | Genome-scale Perturb-seq benchmark and public-screen holdout | Obtain through scPerturb or original publication data links |
| PapalexiSatija2021 ECCITE-seq RNA | Public-screen holdout and case examples | Obtain through scPerturb or original publication data links |
| DatlingerBock2021 | Public-screen holdout | Obtain through scPerturb or original publication data links |
| GSE284197 | Independent public experimental Perturb-seq screen | Obtain from GEO / original data repository |
| GSE264667 / NadigOConner2024 HepG2 and Jurkat | Audited external candidate only, not completed validation evidence | Obtain from GEO if needed; large raw `.h5ad` files are not redistributed here |

## What is included here

- Core Python source code for split generation, baseline contracts, PTL
  training/evaluation, adapter checks, and deterministic figure generation.
- Lightweight manifests that document public-screen summaries, external
  candidate audits, case examples, and method-comparison tables.
- Expected-output documentation.

## What is intentionally excluded

- Raw `.h5ad` files.
- Raw expression matrices and cell-level metadata tables.
- Large prediction arrays such as `.npz` outputs.
- Generated figures and submission packages.
- Cache files such as `__pycache__` and `.pyc`.

## Local data layout

If reproducing the full analysis locally, place downloaded data under a local
`data/` directory that is ignored by git. Scripts assume project-relative paths
for local execution, but these paths are examples only and do not imply that raw
data are bundled with this repository.
