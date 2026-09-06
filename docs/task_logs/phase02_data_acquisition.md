# Phase 02 Task Log - Data Acquisition

Date: 2026-04-25
Timezone: Asia/Shanghai

## Scope

Phase 02 acquired and inventoried public single-cell perturbation data from the official scPerturb v1.4 Zenodo record. Work was limited to source verification, dataset download, checksums, inventory files, cleanup of interrupted-download artifacts, and tracking updates. No preprocessing, benchmark split construction, model training, or experiment code was started.

## Skill gate and tools

- Checked `SKILLS_INDEX.md`.
- Read relevant skill documentation for `cellxgene-census`, Life Science Research `research-router-skill`, and `biostudies-arrayexpress-skill`.
- `cellxgene-census` was judged useful for optional reference atlas work but not the primary source for harmonized perturbation `.h5ad` downloads.
- Callable BioStudies/ArrayExpress or NCBI MCP tools were not exposed through tool discovery in this session.
- Main acquisition path used the official scPerturb v1.4 Zenodo record, Zenodo API metadata, direct downloads, and later `zenodo_get`.

## Environment checks

- Preferred environment checked: `E:\anaconda3\envs\sw_mgli`.
- Result: environment exists.
- Python version in preferred environment: 3.9.25.
- `pandas` is installed.
- `zenodo-get` was installed into the preferred environment during acquisition.
- `h5py` and `anndata` are still not confirmed for Phase 03 inspection.

## Source metadata

- Primary source: https://zenodo.org/records/13350497
- API endpoint: https://zenodo.org/api/records/13350497
- Record title: scPerturb v1.4
- Publication date: 2024-08-28
- DOI: 10.5281/zenodo.13350497
- License: CC-BY-4.0

## Download status

- Official scPerturb `.h5ad` files in record `13350497`: 54
- Local `.h5ad` files present after completion: 54
- Local files with exact size match to Zenodo metadata: 54
- Total local `.h5ad` payload: 43,042,767,207 bytes

### Initial verified file

- `NormanWeissman2019_filtered.h5ad` was downloaded and checksum-verified early in Phase 02.
- Logs:
  - `results/logs/downloads/phase02_scperturb_norman_download.log`
  - `results/logs/downloads/phase02_scperturb_norman_download_retry.log`

### Full-record acquisition

- The remainder of the scPerturb record was downloaded into `data/raw/scperturb_v1.4/`.
- The directory now includes the core PTL-relevant files:
  - `NormanWeissman2019_filtered.h5ad`
  - `ReplogleWeissman2022_K562_essential.h5ad`
  - `ReplogleWeissman2022_rpe1.h5ad`
  - `ReplogleWeissman2022_K562_gwps.h5ad`
  - `SrivatsanTrapnell2020_sciplex2.h5ad`
  - `SrivatsanTrapnell2020_sciplex3.h5ad`
  - `SrivatsanTrapnell2020_sciplex4.h5ad`
  - plus all remaining `.h5ad` files in the official record
- `md5sums.txt` is present in the source directory and matches the official Zenodo file list used by `zenodo_get`.

### Cleanup

- Removed 4 stale `.tmp` artifacts from interrupted downloads after the completed `.h5ad` files were confirmed present.
- Cleanup was restricted to `H:\2026try\4.24\data\raw\scperturb_v1.4\*.tmp`.

## Outputs

- Rebuilt `results/tables/data_inventory.csv` as a full 54-file inventory.
- Updated `docs/data_inventory.md` to reflect complete acquisition.
- Confirmed all official `.h5ad` files are present under `data/raw/scperturb_v1.4/`.
- Preserved download logs under `results/logs/downloads/`.
- Updated project tracking files.

## Acceptance check

- At least one Tier 1 dataset local: passed.
- All official scPerturb `.h5ad` files local: passed.
- Inventory CSV exists: passed.
- Data inventory markdown exists: passed.
- Download logs exist: passed.
- Tracking files updated: passed.
- No Phase 03 work started: passed.

## Open issues

No user intervention is needed for Phase 02. Phase 03 should inspect the downloaded `.h5ad` files after confirming or installing `h5py/anndata` in the preferred environment.
