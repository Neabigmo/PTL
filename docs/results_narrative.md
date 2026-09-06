# Results Narrative

Last synchronized: 2026-09-05 Asia/Shanghai

## Main message

The expanded benchmark supports a context-dependent view of perturbation
transportability. Random-split performance is a useful low-stress anchor, but
the stress families expose different leading baselines, substantial transfer
decay, and concentrated failure modes. PTL is layered over completed
predictions to identify outputs that should be retained or flagged before
downstream biological interpretation.

## Benchmark surface

The active surface contains nine processed public screens: eight scPerturb
datasets and the independent `GSE284197_screen` external holdout. The canonical
signature-track campaign contains 525 rows across five transparent baselines,
three seeds, and six split families. The fixed cross-dataset gene space is the
6,257-gene intersection recorded by the current run matrix. The four additional
public-screen holdouts are `GSE284197_screen`,
`ReplogleWeissman2022_rpe1`, `PapalexiSatija2021_eccite_RNA`, and
`DatlingerBock2021`.

## Benchmark findings

Dataset-heldout transfer is the hardest stress family by average non-control
cosine. The family-level leading rows are `ridge_regression_baseline` for
random, low-support, and unseen-perturbation splits;
`perturbation_mean_delta_baseline` for unseen combinations; and
`global_delta_baseline` for dataset-heldout and external holdout transfer.
The corresponding leading mean non-control cosine values are 0.373, 0.307,
0.311, 0.664, 0.056, and 0.112, respectively. These values describe the
adopted public datasets and implemented baselines; they are not a universal
ordering of all perturbation predictors.

The additional public-screen summaries show heterogeneous transfer behavior:
the mean non-control cosine across five baselines is 0.061 for GSE284197,
0.038 for Replogle RPE1, -0.025 for ECCITE-seq RNA, and 0.006 for
DatlingerBock2021. The best model therefore changes across public screens.

## PTL reliability findings

The primary full deployment-feature random forest reduces mean
false-transportability rate from 0.709 for naive confidence to 0.236, with
ROC AUC 0.929 and average precision 0.926. The strongest comparator is the
calibrated deployment logistic model at 0.268. A separate ablation table shows
that removing context distance gives a similar best-row filter; this is
reported as a robustness check, not as evidence that context information is
unnecessary in general.

The retained/rejected biological summary contains 17,278 retained signatures
with a transportable rate of 0.907, compared with 32,028 rejected high-risk
signatures with a rate of 0.174. Perturbation-retrieval top-1 is 0.086 for
retained signatures and 0.034 for rejected signatures. Pathway summaries are
kept as orthogonal diagnostics rather than treated as uniformly improved
outcomes.

## Failure modes and adapter scope

The failure atlas identifies severe failure, novelty, low support, and
dataset-transfer boundaries as recurring stress conditions. These summaries
are descriptive and threshold-based; they motivate reliability auditing and
follow-up experiments rather than molecular mechanism claims.

The GEARS adapter has three completed output-contract demonstrations and 18
missing planned rows because the optional model dependencies are unavailable
in the preferred environment. GEARS is therefore not a completed performance
comparison in the current evidence line.

## Limitations

The synthesis remains signature-level and public-data based. PTL requires
labeled benchmark outputs and may need recalibration for a new technology,
species, or collection. External transfer is represented by the adopted public
holdouts rather than a universal transfer census, and the failure atlas does
not establish molecular mechanisms or replace experimental validation.
