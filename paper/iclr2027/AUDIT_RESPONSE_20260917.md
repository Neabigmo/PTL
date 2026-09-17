# Audit response — 2026-09-17

This ledger maps the September 17 audit to the current manuscript. It records
what is part of the paper, what is supplementary evidence, and what remains a
declared boundary rather than being overstated.

## Completed in the manuscript

- **Estimand and inference:** `D_adj` is defined as a discrepancy between
  pair-state laws after subtraction of expected within-context disagreement.
  The manuscript now distinguishes the population estimand, its signed
  finite-sample U-statistic estimator, the matched pseudo-context null, and the
  final seed-block interval. Legacy percentile intervals are explicitly marked
  invalid in the Supplement rather than silently reused.
- **Pair universe, ties and support:** the pair weighting, tie kernel, comparator
  universe, tie tolerances, and the minimum strict support of eight Monte Carlo
  outcomes are stated explicitly. The support count is no longer described as
  biological replication.
- **Source-frozen contract:** raw counts, normalization, PCA/ridge fitting,
  label-disjoint out-of-fold predictions, and source-to-target orientation are
  described as one reproducible chain. Target outcomes are reserved for
  evaluation.
- **External evidence:** the main text separates independent Nadig and Tian
  raw-cell replications from predictor-family sensitivity. The Frangieh GEARS
  bridge is reported under the strict source-only contract; incompatible model
  or data combinations are listed as scope boundaries, not positive results.
- **Decision claims:** pooled and within-metric associations are separated.
  The mean-risk-shift control and leave-one-context-pair-out comparison are
  reported with the sign convention made explicit. The claim is deliberately
  limited to shortlist replacement because `D_adj` did not improve held-out
  regret prediction beyond metric identity and mean-risk shift.
- **Narrative:** the paper now has three contributions: reliability transport as
  a problem object, a finite-measurement estimand with calibration, and
  replication plus shortlist consequences. Repeated audit/feasibility material
  was moved out of the main narrative.
- **Terminology:** “measurement depth” was replaced by cell budget where the
  experiment varies cells per perturbation-context pseudoreplicate. Internal
  engineering terms such as “kill-switch” and “materialized” were removed from
  reader-facing tables and prose.
- **Reproducibility and disclosure:** the manuscript uses only the anonymous
  forwarding URL, identifies public raw-data sources, states what is and is not
  redistributed, and includes an AI-use disclosure.

## Figure and layout changes

- The four main figures were regenerated from their current source products.
- Figure 3 is titled and discussed as a **cell-budget** boundary rather than a
  sequencing-depth claim.
- The main paper now ends on page 9; references occupy pages 10–11. This was
  achieved by removing duplicated exposition, not by reducing type size.
- Main and Supplement PDFs compile successfully. The remaining Supplement table
  overflow is confined to a wide legacy calibration table and does not alter
  values; it is retained as a traceable legacy comparison.

## Deliberately bounded or deferred

- **TxPert/scGPT/other external families:** no score is claimed when a compatible
  source-matched checkpoint and legal held-out prediction vector are absent.
- **Feng:** released aggregate data cannot reproduce the raw-cell measurement
  hierarchy, so it remains a neutral external audit rather than a biological
  replication.
- **Prospective utility:** source-only forecasting remains supplementary; the
  current evidence does not support an oracle-like prospective claim.
- **Decision generality:** the 18 rows are not 18 independent biological
  replicates. They arise from three metrics, three unordered Frangieh context
  pairs, and two directions; the limitation is stated in the main Discussion.

## Deliverables

- `main.pdf`: 9 pages of main content plus 2 pages of references.
- `supplement.pdf`: 19 pages of estimator details, calibration, external-model
  constraints, replication ledgers, and robustness analyses.
- Anonymous release: `https://anonymous.4open.science/r/PTL-B377/`.
