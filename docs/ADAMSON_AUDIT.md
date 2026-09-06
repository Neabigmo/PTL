# Adamson 2016 surface audit

Date: 2026-09-06

## Evidence inspected

The read-only H5AD metadata audit inspected all three local Adamson accessions:

| Accession | Cells | Genes | Perturbation categories | Local observation |
| --- | ---: | ---: | ---: | --- |
| GSM2406675 / 10X001 | 5,768 | 35,635 | 9 | K562; small subset; `*` appears as a special perturbation label |
| GSM2406677 / 10X005 | 15,006 | 32,738 | 20 | K562; two explicit negative-control labels; current processed surface has only 10 signatures |
| GSM2406681 / 10X010 | 65,337 | 32,738 | 114 | K562; distinct accession and substantially broader perturbation panel |

All three files expose `perturbation`, `perturbation_type`, `cell_line`,
`celltype`, and QC metadata. They are separate GEO accessions with different
cell counts and perturbation cardinalities. The available local metadata does
not justify treating them as interchangeable technical replicates.

## Decision

The three files are retained as an **Adamson full-surface candidate**, but the
candidate is not headline-eligible yet. We will not concatenate them until the
source study metadata and control semantics establish whether they are
replicates, assay subsets, or distinct experimental conditions. The current
10-signature 10X005 processed artifact remains v1/legacy evidence only.

This prevents a concrete failure mode: silently merging different accessions
could create duplicated biological instances or incompatible controls, causing
optimistic folds and invalid cross-dataset fidelity comparisons. File names,
Git history, and ordinary row IDs cannot resolve that biological meaning; the
audit and an explicit merge policy are required.

## Reproducible candidate configuration

`configs/adamson_full_candidate.yaml` records the three inputs and keeps the
merge policy at `pending_source_metadata_review`. No new Adamson matrix was
written in this audit.
