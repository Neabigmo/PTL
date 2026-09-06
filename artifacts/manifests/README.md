# Tracked manifest contract

Manifests describe dataset versions, environment membership, folds, file
locations and generation commands. They must remain small, repository-relative
and free of credentials or absolute local paths.

Tracked CSV/JSON manifests are the review surface for G1-G4. Large parquet
fold tables and all raw/processed matrices remain ignored and reproducible
from the local data tree.
