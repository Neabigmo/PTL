# Nature-style figure contract

## Figure 1 — framework and audit boundary

Core conclusion: separating native uncertainty from deployment context makes
the reliability problem auditable before any selective decision is made.

Figure archetype: schematic-led composite. Backend: Python/matplotlib.
Target output: double-column manuscript figure, editable SVG/PDF plus 600-dpi
TIFF preview.

Panel map:

- a: predictor output → uncertainty/context blocks → PTL calibrator → keep or
  abstain;
- b: benchmark environments and context expansion status;
- c: GWPS stage-wise coverage audit;
- d: predictions, deployment features, outcomes, and folds kept as separate
  evidence tables.

Evidence hierarchy: the workflow is the hero panel; the GWPS audit and raw
  semantic metadata are validation/decision evidence. No panel uses simulated
  quantitative data.

## Figure 2 — native uncertainty under shift

Core conclusion: native uncertainty is not uniformly aligned with realized
fidelity across split families and predictor families.

Figure archetype: quantitative grid. The main evidence is the observation-level
uncertainty-versus-fidelity relation; aggregate split/predictor summaries are
supporting evidence.

Statistics: points are prediction-level biological instances; split-family and
predictor summaries report the observed mean fidelity and mean uncertainty
quantile. This is a pilot replay, not a final learned-model claim.

## Figure 3 — selective reliability

Core conclusion: target-free PTL features improve selective reliability over a
raw normalized-UQ ordering on the current replay surface.

Figure archetype: quantitative grid. Primary metric is environment-first macro
AURC; paired hierarchical bootstrap intervals are shown where available.
Secondary panels show false-transportability risk and split-family deltas.

## Figure 4 — reliability information blocks

Core conclusion: PTL combines prediction, support, novelty, and biological
context blocks, and the current ablation surface can test whether removing a
block changes selective reliability.

Figure archetype: asymmetric mixed-modality figure. The block schematic is
interpretive; ablation and feature-count panels are quantitative. The full
incremental U→U+P→U+P+S→U+P+S+N→U+P+S+N+C ladder is reserved for the expanded
multi-environment run and is not fabricated from the present ablation table.

## Integrity and review risks

- All quantitative panels read tracked source tables or machine-readable
  manifests and use no random/mock generator.
- Predictor, split, and environment colors are fixed across panels.
- Labels distinguish pilot evidence from final headline claims.
- The current replay uses five grouped folds and environment-first aggregation;
  source tables carry the corresponding sample sizes and interval metadata.
- The GWPS figure explicitly shows why that surface is a GEARS contract pilot,
  not broad headline coverage.
