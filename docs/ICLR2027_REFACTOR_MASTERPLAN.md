# ICLR 2027 PTL-v2 refactor masterplan

This is the local execution summary of the plan agreed in the PTL project
chat. The detailed discussion remains in that chat; this file records the
source-of-truth decisions needed by the repository.

## Scientific spine

The central question is: **How much reliability information remains in
biological context after predictor uncertainty is known?** The paper has three
separate layers:

1. A predictor produces a response and uncertainty.
2. Predictor-relative normalization and scalar calibration produce a baseline
   confidence.
3. PTL uses prediction geometry, support, novelty, biological context and
   predictor descriptors to correct that confidence before keep/abstain.

The main method preserves confidence ordering within a context:

`p = sigmoid(exp(a(z)) * logit(p0) + b(z))`

with regularization toward `a=0, b=0`. A continuous-risk head is implemented as
an ablation and retained only if it improves grouped out-of-fold results.

## Data and identity

- Common Core: Norman, Replogle K562, Replogle RPE1 and fully audited Adamson.
- Context Expansion: Tian CRISPRa/CRISPRi, Frangieh, Papalexi, Xu and possibly
  Datlinger, selected for biological context diversity rather than count.
- External: retain GSE284197 as External-A with transparent prior use; select
  External-B from Wessels/Joung using a pre-result rubric and freeze it before
  PTL evaluation.
- Separate model gene space from a shared evaluation gene space.
- Define environment ID, biological-instance ID and prediction ID. PTL folds
  group by biological instance, not run ID.

## Contracts and evaluation

Keep `predictions`, `deployment_features`, `outcomes` and `folds` separate.
Prediction/UQ artifacts cannot contain fidelity outcomes; PTL joins them only
at evaluation time. Report environment macro-averages with hierarchical
confidence intervals. The primary metrics are selective risk (AURC/eAURC,
FTR@50/80%, Risk@80%), then calibration, discrimination and continuous risk.
Expression delta cosine, DEG direction consistency and perturbation retrieval
are the three main biological views; pathway metrics remain supplementary.

## Research questions and paper surface

- RQ1: Does uncertainty fail conditionally across support, novelty, context and
  screen shifts?
- RQ2: Can contextual calibration repair uncertainty relative to raw UQ,
  scalar calibration, RF/GBDT and PTL-Context baselines?
- RQ3: What is learned by cumulative U → U+P → U+P+S → U+P+S+N → Full blocks?
- RQ4: How robust is the correction under held-screen, held-predictor,
  alternative metric, high-quality subset and External-B checks?

The new paper lives under `paper/iclr2027/` and is not a conversion of the
CBAC manuscript. Four main figures cover the question, contextual confidence,
PTL correction and information/robustness; v1 audit, failure atlas and case
assets move to supplementary evidence only after G6.

## Execution order

G0 Git/bootstrap → G1 data registry → G2 predictor contract → G3 uncertainty
contract → G4 small PTL pilot → G5 predictor expansion → G6 full experiments →
G7 ICLR paper. G0 deliberately does not run new experiments.
