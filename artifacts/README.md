# Artifact policy

Generated artifacts are ignored by default. Only small, reviewable manifests
and compact source-data tables may be selectively tracked under
`artifacts/manifests/` and `artifacts/source_data/`.

Raw data, checkpoints, giant prediction tables and reproducible build output
remain outside Git and are referenced by manifests and generation commands.
