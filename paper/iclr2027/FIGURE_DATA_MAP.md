# Figure data map

Every submission panel is traceable to a canonical artifact and filter below.

| Figure | Panel | Source artifact | Columns / fields | Filtering | Aggregation |
|---|---|---|---|---|---|
| Fig.1 | A | frozen Frangieh protocol | context labels; predictor status | canonical setup | schematic only |
| Fig.1 | B | frangieh_source_frozen_predictions.npz + canonical Frangieh truth registry | perturbation label; source/target risk; rank | source-rank quantile exemplars from the full 243-label surface | display risk rank; full surface remains the source table |
| Fig.1 | C | formal_v2_claim_lock_measurement_fullsize_summary.csv | cross disagreement; measurement floor; identifiable divergence; CIs | primary Ctrl↔IFNγ; full depth | none |
| Fig.1 | D | Frangieh source-frozen prediction product | source/target shortlist counts; complete label accounting | top 10% of the 243-label surface; compact preview | 6 retained; 19 dropped; 19 new; full source table retained |
| Fig.2 | A | reliability_transport_atlas.csv | source; contrast; metric; measurement identifiable; CI | Frangieh 3×3×3 canonical cells | none |
| Fig.2 | B | stable_inversion_summary.csv | source direction; metric; strict inversion fraction | six directed Frangieh transfers; strict support ≥8 | stable pair reversal fraction |
| Fig.2 | C | reviewer_pseudocontext_negative_control.csv | same-context source; metric; measurement identifiable | three contexts × three metrics | point estimates only |
| Fig.2 | D | formal_v2_claim_lock_replication_nadig_fullsize.csv | source direction; metric; measurement identifiable; CI | Nadig two directions × three metrics | direction-specific point and 90% interval |
| Fig.3 | A | reliability_transport_measurement_depth_matched_fixed_summary.csv | source; contrast; metric; depth; identifiable divergence; CI | all canonical contrasts | depth trajectory |
| Fig.3 | B | reliability_transport_measurement_depth_matched_fixed_resolution.csv | source; contrast; metric; thresholds | all source×contrast cells | threshold glyph |
| Fig.3 | C | finite_measurement_rank_discrimination.csv | comparator; depth; cross estimate; same-context null; margin; CI | five comparator families; 18 directed surfaces; 10/20/40/80/160/full | cross-versus-matched-null discrimination |
| Fig.4 | A | Frangieh source-frozen prediction product | perturbation label; source/target rank; selection flags | all 243 shared labels; top 10% | shortlist replacement |
| Fig.4 | B | reliability_transport_decision_link.csv | transfer; metric; d_meas_id; normalized regret; CIs | full depth; 6 transfers×3 metrics | none |
| Fig.4 | C | reviewer_decision_incremental_value.csv | nested model; OOF MAE/RMSE; transfer-bootstrap interval; exact block permutation | five predeclared models; six transfer LOTO | incremental decision value beyond metric identity |
| Supplementary Fig.S1 | A | reviewer_prospective_feature_ladder.csv | legal source-only feature ladder | target leakage false | information firewall |
| Supplementary Fig.S1 | B | reviewer_prospective_feature_ladder.csv | task; feature ladder; Spearman | 18 continuous-burden tasks per ladder | median and IQR |
| Supplementary Fig.S1 | C | reviewer_prospective_feature_ladder.csv | task; feature ladder; AUPRC | 18 high-burden tasks per ladder | median and IQR |
| Supplementary Fig.S2 | A | reliability_transport_decision_link.csv | 18 fixed transfer×metric decision rows; D_adj; regret; intervals | full depth; 10% random-reference budget | native-scale decision surface |
| Supplementary Fig.S2 | B | reviewer_decision_metric_stratified.csv/json | within-metric ranks; metric-specific and pooled Spearman | six rows per metric; unordered context-pair block bootstrap | metric-stratified association |
| Supplementary Fig.S3 | A–B | metric_matched_mean_fidelity_control.csv | metric-specific mean-risk shift; D_adj; normalized regret | exact matched-fixed label universes; 18 rows | same-metric fidelity versus ordering/decision |
| Supplementary Fig.S4 | A–B | ordering_identifiability_synthetic_trials.csv | trial-level plug-in and U-corrected estimands | canonical finite null and known alternative | stored synthetic sampling distributions |
| Supplementary Fig.S5 | A–D | simulation_v21/finite_measurement_primary_v21.csv + predictor_families_v2/canonical_four_family_transport.csv | finite-truth calibration; depth; resolution; noise law; inversion; estimator; predictor family; source fidelity | 960 preregistered primary conditions; 500 trials/condition; four source-only families | finite-measurement calibration and canonical family sensitivity |

Repository-only auxiliary analyses (condition holdout, failure anatomy, and the former broad deployment gate) are not manuscript figures. Global contracts: metric colors are fixed; unavailable cells remain NA/gray; strict states require support ≥8; no target-informed feature enters Supplementary Fig. S1; no mock numerical values are used. Supplementary Fig. S5 uses finite-depth population truth and canonical raw-cell family surfaces; it does not replace the primary estimator or imply family superiority.
