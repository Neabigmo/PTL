# External context replication: current decision

The outcome-blind audit of SrivatsanTrapnell2020_sciplex3.h5ad is recorded in
artifacts/manifests/sciplex3_context_replication_audit_v2.json.

The file contains 189 shared perturbations and 753 matched dose-by-time units
for each of A549/MCF7/K562 pair, with median minimum cell counts of 221, 213,
and 241 respectively. Controls are present in all three contexts and the
readout uses one fixed gene panel. However, all 52 observed plates are unique
to a single cell line, so the registered no-unique-plate-confounding criterion
fails. The source-only predictor was therefore not run on this external file.

To reopen this audit, a provenance-preserving external context replicate must
provide either plate-balanced acquisition across the three cell lines or an
explicit batch design that supports a pre-registered plate-aware analysis. The
file must retain matched perturbation, dose, time, replicate, control, and
raw-cell metadata. Expression outcomes must remain unavailable to the
outcome-blind eligibility decision.
