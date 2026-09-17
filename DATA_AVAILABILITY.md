# Data availability

The underlying single-cell data are public but are not redistributed in this
repository because the matrices are large and remain subject to the source
dataset terms. The scPerturb v1.4 Zenodo record is the canonical source:

- Record: https://zenodo.org/records/13350497
- DOI: `10.5281/zenodo.13350497`
- Source collection: scPerturb single-cell perturbation RNA/protein H5AD files

The primary manuscript analysis uses these files from that record:

- `FrangiehIzar2021_RNA.h5ad` (Frangieh Perturb-CITE-seq RNA screen)
- `NadigOConner2024_hepg2.h5ad` (Nadig HepG2 screen)
- `NadigOConner2024_jurkat.h5ad` (Nadig Jurkat screen)

The repository contains repository-relative configuration templates and compact
summary/source products used to inspect the reported figures. It does not
contain H5AD/NPZ matrices, cell-level metadata, model checkpoints, or local
filesystem paths. Downloaded data should be stored only in the ignored data
directories specified by the configuration files.

The strict GEARS analysis can be regenerated from the public Frangieh matrix
with the repository scripts. This anonymous release provides only de-identified
task-level transport summaries and calibrated intervals; trained weights and
cell-level prediction arrays are intentionally omitted.
