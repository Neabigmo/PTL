# Current Data Inventory

Last synchronized: 2026-09-05 Asia/Shanghai

## Primary source

- Source: scPerturb v1.4 on Zenodo
- Record: https://zenodo.org/records/13350497
- DOI: 10.5281/zenodo.13350497
- Published: 2024-08-28
- License: Creative Commons Attribution 4.0 International
- Local root: `H:\2026try\9.5PTL\data\raw\scperturb_v1.4`

## Download status

- Official Zenodo `.h5ad` files in this record: 54
- Local `.h5ad` files present: 54
- Size-matched against Zenodo metadata: 54 / 54
- Total downloaded payload: 43,042,767,207 bytes
- Residual interrupted-download `.tmp` files: removed

## Key files now available

| Dataset family | Representative local files | Why they matter for PTL |
|---|---|---|
| Core Perturb-seq transport set | `NormanWeissman2019_filtered.h5ad`, `ReplogleWeissman2022_K562_essential.h5ad`, `ReplogleWeissman2022_rpe1.h5ad`, `ReplogleWeissman2022_K562_gwps.h5ad` | Covers combinatorial CRISPRa, essential-gene CRISPRi, cross-cell-line transport, and genome-wide perturbation stress testing. |
| Chemical perturbation | `SrivatsanTrapnell2020_sciplex2.h5ad`, `SrivatsanTrapnell2020_sciplex3.h5ad`, `SrivatsanTrapnell2020_sciplex4.h5ad` | Gives non-CRISPR perturbation settings and dose/context variation. |
| Regulatory / enhancer screens | `GasperiniShendure2019_lowMOI.h5ad`, `GasperiniShendure2019_highMOI.h5ad`, `GasperiniShendure2019_atscale.h5ad` | Useful for perturbation classes beyond standard gene knockdown. |
| In vivo / ex vivo / disease context | `LaraAstiasoHuntly2023_invivo.h5ad`, `LaraAstiasoHuntly2023_exvivo.h5ad`, `LaraAstiasoHuntly2023_leukemia.h5ad`, `McFarlandTsherniak2020.h5ad` | Important for context-stress and transportability across biological settings. |
| Multimodal perturbation | `PapalexiSatija2021_eccite_RNA.h5ad`, `PapalexiSatija2021_eccite_protein.h5ad`, `FrangiehIzar2021_RNA.h5ad`, `FrangiehIzar2021_protein.h5ad` | Supports reliability analysis across modality-specific readouts. |

## Current processed benchmark surface

The latest preprocessing contract contains nine processed datasets: eight
scPerturb datasets and the active external `GSE284197_screen`. The processed
dataset summary is recorded in `results/tables/preprocessing_summary.csv`.

| Dataset | Role | Kept cells | Signature rows | Unique perturbations |
|---|---|---:|---:|---:|
| NormanWeissman2019_filtered | scPerturb train | 110,772 | 1,829 | 237 |
| ReplogleWeissman2022_K562_essential | scPerturb train | 51,821 | 3,258 | 463 |
| GSE284197_screen | independent external holdout | 51,728 | 73 | 46 |
| AdamsonWeissman2016_GSM2406677_10X005 | scPerturb train | 14,957 | 10 | 10 |
| DatlingerBock2017 | scPerturb train | 3,685 | 179 | 63 |
| DixitRegev2016_K562_TFs_7_days | scPerturb train | 27,925 | 11 | 11 |
| PapalexiSatija2021_eccite_RNA | scPerturb train | 20,517 | 266 | 93 |
| ReplogleWeissman2022_rpe1 | scPerturb train | 43,715 | 1,797 | 166 |
| DatlingerBock2021 | scPerturb train | 5,373 | 81 | 41 |

The active external validation source is `GSE284197_screen` from GEO because
it exposes reproducible per-cell perturbation labels, controls, and context
fields. `E-MTAB-14567` remains locally archived but is not active validation
evidence because its per-cell perturbation mapping did not meet the adopted
contract.

## Historical assessment

- For this project, the current scPerturb download is already enough to support Phase 03 through the main benchmark-building phases.
- The external validation lane has been upgraded beyond the earlier BioStudies candidate.
- The active external validation source is now `GSE284197_screen` from GEO because it exposes reproducible per-cell perturbation labels, controls, and context fields.
- `E-MTAB-14567` is retained locally as an archived candidate but is not part of the active external-validation contract.

## External validation extension

### Active external validation dataset: GSE284197_screen

- Source: GEO supplementary files
- Accession: `GSE284197`
- URL: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE284nnn/GSE284197/suppl/
- Local root: `H:\2026try\9.5PTL\data\external\GSE284197`
- Primary file: `GSE284197_screen.h5ad`
- Downloaded file size: `4,450,088,782` bytes

### Why it was adopted

- The backed `.h5ad` is readable with shape `137,317 x 38,593`.
- `obs` includes direct per-cell perturbation and context fields:
  - `perturbation`
  - `sgRNA`
  - `sgRNA_effective`
  - `Gene_target`
  - `Gene_target_single`
  - `batch`
  - `timepoint`
  - `species`
  - `stage`
- `species` is uniformly `human`.
- Control labels are directly recoverable from `perturbation` as `WT` and `NT`.
- Single-target perturbations are recoverable from `Gene_target_single`.

### Archived candidate: E-MTAB-14567

- Source: BioStudies ArrayExpress
- Accession: `E-MTAB-14567`
- URL: https://ftp.ebi.ac.uk/pub/databases/biostudies/E-MTAB-/567/E-MTAB-14567/
- Release date: 2025-04-12
- Local root: `H:\2026try\9.5PTL\data\external\E-MTAB-14567`
- Downloaded bundle size: 1,000,378,417 bytes

### Files acquired

| File | Size | Role |
|---|---:|---|
| `E-MTAB-14567.idf.txt` | 7,751 | Study-level metadata |
| `E-MTAB-14567.sdrf.txt` | 65,328 | Sample / assay metadata |
| `filtered_feature_bc_matrix.h5` | 29,971,316 | Original 10x matrix including 491 CRISPR Guide Capture features |
| `filtered_feature_bc_matrix_rerun.h5` | 145,197,886 | Rerun 10x matrix with gene-expression features only |
| `molecule_info.h5` | 825,136,136 | Molecule-level raw support file |

### Inspection result

- `filtered_feature_bc_matrix.h5` is readable with shape `35,260 x 20,354`.
- `filtered_feature_bc_matrix.h5` contains:
  - `19,863` gene-expression features
  - `491` CRISPR guide-capture features
- `filtered_feature_bc_matrix_rerun.h5` is readable with shape `34,900 x 19,863`.
- `filtered_feature_bc_matrix_rerun.h5` contains gene-expression features only.
- The bundle remains useful as an archived candidate, but it is **not part of the active Phase 03 external-validation path** because the current files do not expose a clean, ready-to-use per-cell perturbation mapping contract.

## Notes

- `results/tables/data_inventory.csv` has been rebuilt as a full 54-file inventory synced to the Zenodo record.
- `results/tables/data_inventory.csv` now includes both the archived BioStudies candidate `E-MTAB-14567` and the adopted GEO external-validation dataset `GSE284197_screen`.
- Preferred conda environment `E:\anaconda3\envs\sw_mgli` now includes the required preprocessing stack for Phase 03.
