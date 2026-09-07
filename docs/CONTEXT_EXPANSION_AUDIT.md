# Context expansion audit

Date: 2026-09-06
Scope: GWPS coverage and Tian/Frangieh biological-context semantics

## Decision summary

The Replogle K562 GWPS surface is retained as a GEARS contract pilot but is
not used as the headline common-core environment. Its processed coverage is
too sparse after batch-aware perturbation eligibility: 1,989,578 raw cells
and 9,866 non-control labels produce 77,168 retained cells, 31 retained
non-control labels, and 440 reference groups. The loss is not explained by
label parsing or cell-level QC; it is dominated by the minimum-cell rule for
batch-by-perturbation groups.

The headline common-core fallback is the Replogle K562 essential surface.
GWPS remains scientifically useful for validating the official GEARS adapter,
native log-variance uncertainty, condition-disjoint splitting, and matched-
reference delta semantics, but its pilot must not be presented as broad GWPS
benchmark coverage.

## GWPS stage audit

| Stage | Cells / groups | Interpretation |
|---|---:|---|
| Raw observations | 1,989,578 cells | 9,866 non-control labels; 75,328 controls |
| Label parsing | 1,989,578 cells | No loss |
| Contract filter | 1,989,578 cells | No loss |
| Minimum genes | 1,988,532 cells | Negligible loss |
| Minimum counts | 1,986,836 cells | Negligible loss |
| Maximum mitochondrial fraction | 1,986,836 cells | No additional loss |
| Batch × perturbation eligibility | 77,168 cells | 31 non-control labels; 440 reference groups |
| Reference matching | 440 signature groups | 267 control signature rows |

The machine-readable audit is stored in
`artifacts/manifests/gwps_coverage_audit.csv` and
`artifacts/manifests/gwps_coverage_audit.json`.

## Semantic corrections

Raw metadata-only audits establish the following meanings before any model
comparison:

- Tian CRISPRa: 21,193 cells from iPSC-induced neurons, 101 perturbations.
- Tian CRISPRi: 32,300 cells from iPSC-induced neurons, 185 perturbations.
- Frangieh RNA: 218,331 cells from melanoma co-culture, with `perturbation_2`
  values `Control`, `Co-culture`, and `IFNγ`; 249 perturbations.

Thus Tian is not a K562 environment, and Frangieh is not a THP-1 environment.
The registry records these contexts explicitly. The source-level evidence is
available in `artifacts/manifests/context_semantics_audit.csv` and
`artifacts/manifests/context_semantics_audit.json`.

## Preprocessing status

The Tian surfaces have completed the current staged pipeline:

- Tian CRISPRa: 19,501 cells retained and 202 signatures generated.
- Tian CRISPRi: 30,427 cells retained and 739 signatures generated.

Frangieh is processed from the raw H5AD with `perturbation_2` retained as a
delta-reference field: 216,931 cells retained and 737 signatures generated.
The condition-stratified signature counts are Control 245, Co-culture 245,
and IFNγ 247; each condition contributes a matched control profile. These
counts are read from the generated preprocessing summary and signature table,
not estimated from the raw metadata audit.

## Reproducibility

The audits are metadata-first and do not load the large expression matrix for
the GWPS or semantic checks. The staged preprocessing configs are:

- `config/preprocessing/tian_kampmann_2021_crispra.yaml`
- `config/preprocessing/tian_kampmann_2021_crispri.yaml`
- `config/preprocessing/frangieh_izar_2021_rna.yaml`

The active task registry places Tian and Frangieh in context expansion and
places GWPS in `gears_contract_pilots`; only K562 essential is selected for
the headline common matrix at this stage.

## Formal context-v2 surface

The formal benchmark surface is now materialized by
`configs/context_benchmark.yaml` and `scripts/build_context_benchmark.py`.
It contains eight explicit environments: Norman K562, Replogle K562
essential, Replogle RPE1, Tian CRISPRa, Tian CRISPRi, and three Frangieh
conditions (Control, Co-culture, and IFNγ). Frangieh conditions are separate
environment rows and use the raw `perturbation_2` field.

The v2 evaluation panel is the exact intersection of the six source feature
universes after explicit whitespace normalization, with 6,897 shared genes.
Predictor-native training spaces remain separate from this comparison panel;
the panel is not a claim that every predictor supports every environment.

The predictor-by-environment matrix in
`artifacts/manifests/predictor_environment_coverage.csv` is planning metadata
only. It enumerates the six-predictor roster and 48 planned cells, while
learned-model outputs, uncertainty estimates, and transfer metrics remain
pending until the corresponding runs complete.
