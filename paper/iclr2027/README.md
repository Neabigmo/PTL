# ICLR 2027 paper boundary

The ICLR version gets a new narrative and source tree. Do not transform the
active CBAC files under `manuscript/` into the ICLR manuscript. Paper sources,
figures and tables are maintained here and must trace to the active PTL-v2
artifacts under `artifacts/source_data/` and `artifacts/manifests/`.

`main.tex` is the current science-first draft using the user-provided ICLR
2027 style files vendored under `third_party/iclr2027/`. Its quantitative
claims are explicitly limited to the validated PTL replay and its figures are
generated from the tracked source tables. The canonical package currently
contains four main figures, Supplementary Figs. S1--S5, and a scope-limited
public-cohort audit covering GEARS plus five matched baseline families. The
Frangieh source-only sensitivity now includes the canonical bilinear, RBF,
direct-MLP and latent-response-MLP surfaces under one raw-cell measurement
contract. This does not promote the roster to a cross-dataset claim: the
scPerturb3 context expansion is retained as an outcome-blind audit and is
blocked because every plate is unique to one cell line. The author block
remains a release-time field.
