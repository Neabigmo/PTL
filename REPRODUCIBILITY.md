# Reproducibility

## Scientific regeneration path

The manuscript is regenerated from canonical CSV/JSON manifest products,
fixed source-frozen splits, declared resampling settings, and the figure
builders in `scripts/`. The scientific submission comprises four main figures,
Supplementary Fig. S1, and compact supplementary tables. Run the main figure
builder followed by `scripts/build_supplement_tables.py`, then compile
`paper/iclr2027/main.tex` and `paper/iclr2027/supplement.tex`. The figure data
map records the artifact and filter for every submission panel.

## Corrected finite-measurement calibration

The corrected primary calibration uses the finite-depth pair-state population
as the truth for $D_{\rm adj}$; it does not use latent Kendall distance as a
surrogate. The 960-condition grid is split into four non-overlapping shards so
long runs can be resumed without mixing partial outputs:

```text
E:\\anaconda3\\envs\\pytorch-clean\\python.exe scripts/run_finite_measurement_simulation_v21.py --mode primary --trials-per-condition 500 --workers 1 --truth-draws 1024 --shard-index 0 --shard-count 4
E:\\anaconda3\\envs\\pytorch-clean\\python.exe scripts/merge_finite_measurement_v21.py --mode primary --shard-count 4
E:\\anaconda3\\envs\\pytorch-clean\\python.exe scripts/build_v21_supplement_table.py
```

Repeat the first command with shard indices 1, 2 and 3. The merged artifact is
`artifacts/manifests/simulation_v21/finite_measurement_primary_v21.csv`; each
condition has 500 trials, six depths, four resolution settings, four noise
scales, five inversion settings and one of two declared noise laws. The
canonical estimator contract is documented in
`paper/iclr2027/ESTIMATOR_CONTRACT_V2.md`.

## Source-only family sensitivity

The neural family predictions are source-only and are evaluated on the same
canonical raw-cell measurement engine as the bilinear and RBF families. The
full-size run is deliberately split into six contiguous, non-overlapping
five-seed shards per family; each shard writes independent bootstrap inputs
before the final synchronized merge:

```text
$py = 'E:\\anaconda3\\envs\\pytorch-clean\\python.exe'
$chunks = @(@(0,5), @(5,10), @(10,15), @(15,20), @(20,25), @(25,30))
foreach ($family in @('source_only_mlp','source_only_latent_mlp')) {
  $prediction = 'artifacts/source_data/' + $family + '_predictions.npz'
  foreach ($chunk in $chunks) {
    $output_stem = $family + '_fullsize_part' + ('{0:d2}' -f ($chunk[0]/5+1))
    & $py scripts/run_family_measurement_chunk.py --family $family `
      --prediction-path $prediction `
      --output-stem $output_stem `
      --draws 2000 --seed-start-index $chunk[0] --seed-stop-index $chunk[1]
  }
  $part_stems = 1..6 | ForEach-Object { $family + ('_fullsize_part{0:d2}' -f $_) }
  & $py scripts/merge_claim_lock_measurement_chunks.py `
    --part-stems $part_stems `
    --output-stem ($family + '_measurement_fullsize')
  & $py scripts/materialize_family_canonical.py --family $family `
    --measurement-report ('artifacts/manifests/' + $family + '_measurement_fullsize.json')
}
& $py scripts/build_canonical_four_family_transport.py
& $py scripts/build_canonical_family_supplement_table.py
```

The checked-in canonical outputs contain 18 rows per family (six directed
transfers by three metrics), 30 measurement seeds, two independent
with-replacement replicates per seed, and minimum strict support eight.
The scPerturb3 expansion is intentionally not run: its outcome-blind audit
found all 52 plates to be unique to one cell line, failing the declared
no-unique-plate-confounding requirement. See
`artifacts/manifests/sciplex3_context_replication_audit_v2.json`.

## Auxiliary analyses not used for claims

The external condition-holdout experiment, old failure-anatomy graphics, and
the earlier broad source-only deployment gate remain available as repository
artifacts. They are not included in the submission PDF and must not be treated
as support for the reliability-transport claim.

## Runtime compatibility ledger

This ledger records bounded engineering outcomes, not biological conclusions.

| Predictor family | Runtime/protocol status | Manuscript consequence |
|---|---|---|
| State | Official checkout and preprocessing available; no exact frozen source-only vector was materialized. | Not pooled with transport estimates. |
| TxPert | Cached K562 inference does not match the Nadig HepG2 source. | Not promoted to the Nadig claim. |
| scGPT | Local torchtext extension import was incompatible. | No negative biological inference. |
| GEARS | Vendored import did not meet the declared local compatibility boundary. | Retained only as engineering provenance. |
