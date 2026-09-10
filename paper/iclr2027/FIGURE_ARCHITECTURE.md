# Figure architecture

This document freezes the compositional plan for the six main figures. The
uploaded six-panel images are visual references only; every quantitative mark
is generated from the canonical PTL artifacts listed in the source tables.

## Visual grammar

* Metric identity is stable across all figures: `D_cross` / delta cosine is
  blue, Systema centroid is orange, and absolute-effect rank is green.
* Biological context is encoded by position, labels, and line style; it never
  reuses a metric color. State is encoded separately: stable directions use
  blue/red, ties use neutral gray, and unstable support is light gray.
* Floor semantics are explicit: cross disagreement is dark charcoal,
  measurement floor is light gray, joint/model floor is medium gray, and the
  identifiable component uses the metric color.
* Every figure has one visually dominant hero relationship, a secondary
  distribution or validation view, and a compact aggregate summary. No figure
  is an equal-weight 2x3 grid and no heatmap is allowed to carry the story.

## Figure compositions

```text
FIG 1  PROBLEM / GRAPHICAL ABSTRACT
┌──────────────────────────────────────────────────────────────────────────┐
│ source-frozen contract (small)       state legend + depth cue (support)   │
├──────────────────────────────────────────────────────────────────────────┤
│                 HERO: continuous source → co-culture → IFNγ rank braid    │
│                 real perturbation labels; 10 deterministic exemplars      │
├───────────────────────────────┬──────────────────────────────────────────┤
│ nested observed = cross +      │ measurement depth → D_meas-ID → top-k   │
│ noise + meas-ID decomposition   │ consequence (compact, real values)       │
└───────────────────────────────┴──────────────────────────────────────────┘
         hero 55% · support 25% · summary 20%

FIG 2  ATLAS
┌──────────────┬─────────────────────────────────────────────────────────────┐
│ rank braid   │ HERO: context-transfer glyph atlas (metric columns)       │
│ (small)      │ cells carry effect geometry + identifiability symbols     │
├──────────────┴───────────────────────────────┬─────────────────────────────┤
│ burden distributions across perturbations     │ Nadig replication (small)  │
│ + compact joint-identifiability annotation    │ unavailable is explicit     │
└───────────────────────────────────────────────┴─────────────────────────────┘
         hero atlas 52% · rank/support 23% · distributions 25%

FIG 3  BOUNDARY
┌──────────────────────────────────────────┬────────────────────────────────┐
│ HERO: signed D_meas-ID trajectories       │ narrow threshold strip:        │
│ depth × source/contrast/metric, 90% CI    │ detect / resolve / unresolved   │
│ with actual state markers                 │ per source × contrast           │
├──────────────────────────────────────────┴────────────────────────────────┤
│ six ridge/raincloud distributions (not histogram) + nested floor legend   │
└──────────────────────────────────────────────────────────────────────────┘
         hero trajectories 62% · threshold strip 18% · distributions 20%

FIG 4  DECISION
┌──────────────┬─────────────────────────────────────────────────────────────┐
│ top-k        │ HERO: 18-point full-depth D_meas-ID → normalized regret    │
│ replacement  │ rho=0.7998, q05=.7388, q95=.9003; marginal/rug + claim     │
├──────────────┴──────────────────────────────────────────────┬──────────────┤
│ four-budget retention/regret trajectories with arrows       │ decomposition│
│ Pareto interpretation; source/target retained/dropped/new    │ compact      │
└──────────────────────────────────────────────────────────────┴──────────────┘
         hero link 48% · alluvial 22% · budgets/Pareto 22% · decomp 8%

FIG 5  FAILURE ANATOMY
┌──────────────────────┬──────────────────────┬──────────────────────────────┐
│ q10 rank glyph       │ q50 rank glyph       │ q90 rank glyph               │
│ low burden           │ median burden        │ high burden                  │
├──────────────────────┴──────────────────────┴──────────────────────────────┤
│ HERO: state alluvial (stable / inversion / tie / unresolved)               │
├─────────────────────────────────────────────────────────────────────────────┤
│ response displacement vs burden: hexbin/KDE + raw points + 3 metric dots   │
│ actual rho by metric; 219/2784 coverage as annotation, full bar in Supp.   │
└─────────────────────────────────────────────────────────────────────────────┘
         case cards 35% · state hero 28% · relation 37%

FIG 6  PROSPECTIVE FIREWALL
┌──────────────────────┬─────────────────────────────────────────────────────┐
│ HERO: asymmetric     │ performance landscape: 2 source-only models ×       │
│ source-only firewall │ 3 metrics; Spearman and AUROC encoded together       │
│ allowed vs forbidden  │ real negative results retained                      │
├──────────────────────┴──────────────────────────────────────┬──────────────┤
│ task-level beeswarm/strip with zero/chance references          │ Nadig       │
│ (no heatmap)                                                  │ blocked      │
└──────────────────────────────────────────────────────────────┴──────────────┘
         firewall 40% · performance 40% · task strip 15% · limitation 5%
```

## Execution order and visual QA

The figures are rendered and inspected in this order: **4 → 5 → 1 → 3 → 2 →
6 → manuscript**. For each figure the PNG and compiled PDF page are inspected
at final size before the next figure is changed. The checklist in
`FIGURE_VISUAL_QA.md` is marked PASS only after that inspection.

The scientific contract is unchanged: frozen Frangieh/Nadig inputs, the three
metrics, 30 seeds, three model members, minimum support 8 for strict states,
2000 decision bootstrap draws, and source-only prospective prediction.
