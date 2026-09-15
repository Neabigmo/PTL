# Estimator contract v2

This note fixes the sampling unit used by the finite-measurement reliability
analysis. It is an implementation contract, not a new runtime gate.

## Objects

For context \(e\), perturbation label \(p\), frozen model member \(m\), raw-cell
measurement seed \(s\), and independent measurement replicate \(h\in\{0,1\}\),
let

\[
R_{p,e,m,s,h}
\]

denote the metric-specific risk of member \(m\)'s frozen prediction against the
raw-cell truth pseudoreplicate. The primary fixed-predictor risk is computed
from the mean of the three frozen members,

\[
\bar R_{p,e,s,h}=\frac{1}{3}\sum_{m=1}^{3}R_{p,e,m,s,h}.
\]

For every unordered perturbation pair \(p<q\), define the categorical state

\[
Z_{pq,e,s,h}=\operatorname{sign}
(\bar R_{p,e,s,h}-\bar R_{q,e,s,h})
\in\{-1,0,+1\}.
\]

The same state definition is used for the individual-member joint-floor
diagnostic, replacing \(\bar R\) by \(R_m\).

## Primary estimator

The submitted full-size table uses 30 prespecified measurement seeds and two
independent with-replacement raw-cell pseudoreplicates per seed. For one
source-trained predictor and one unordered context pair, the implementation
concatenates the 60 vectors \((s,h)\) within each context. It then computes

\[
\widehat C_{st}
=\frac{1}{H_sH_t}\sum_{a=1}^{H_s}\sum_{b=1}^{H_t}
\frac{1}{P}\sum_{p<q}
\mathbf{1}\{Z_{pq,s}^{(a)}\ne Z_{pq,t}^{(b)}\},
\]

and the order-2 within-context U estimates

\[
\widehat W_e^U
=\binom{H_e}{2}^{-1}\sum_{a<b}
\frac{1}{P}\sum_{p<q}
\mathbf{1}\{Z_{pq,e}^{(a)}\ne Z_{pq,e}^{(b)}\}.
\]

The reported measurement-adjusted divergence is

\[
\widehat D_{\mathrm{adj}}
=\widehat C_{st}
-\frac{\widehat W_s^U+\widehat W_t^U}{2}.
\]

Here \(H_s=H_t=60\), \(P=\binom{n_{\mathrm{labels}}}{2}\), and only labels that
pass the matched full-size minimum-cell rule enter the common universe. The
source predictor is frozen before target outcomes are loaded; target outcomes
are never used for fitting or tuning.

## Proposition 1: finite-measurement unbiasedness

Conditionally on the frozen predictor and a fixed perturbation universe, if the
measurement pseudoreplicates are independent draws from context-specific
pair-state laws \(\pi_s\) and \(\pi_t\), then

\[
E[\widehat W_e^U]=1-\|\pi_e\|_2^2
\]

and

\[
E[\widehat D_{\mathrm{adj}}]
=\frac12\|\pi_s-\pi_t\|_2^2
=D_{\mathrm{adj}}^\star.
\]

The result is pairwise over the three categorical states and does not treat
the \(P\) correlated perturbation pairs as independent bootstrap units. The
implementation's perturbation-label bootstrap resamples labels, while the
measurement-seed selection is retained as a separate declared uncertainty
component.

## Proposition 2: deterministic tie-free reduction

If each context has a deterministic, tie-free ordering, every \(\pi_e\) is a
point mass for each pair. Therefore

\[
D_{\mathrm{adj}}^\star
=\frac{1-\tau}{2},
\]

where \(\tau\) is Kendall's rank correlation for the two deterministic
orderings. This identity is not used as the finite-measurement truth when ties
or sampling noise are present.

## Signed finite estimates

The population quantity \(D_{\mathrm{adj}}^\star\) is non-negative, but the
finite U-corrected estimate is signed and is not truncated. A negative observed
value means that the finite measurement sample produced a downward fluctuation;
it is not evidence that the population squared distance is negative.

## Distinct diagnostics

1. **Full-size primary surface.** The 30-seed × 2-replicate flattened estimate
   above is the estimator used in the canonical full-depth family outputs.
2. **Split-half diagnostic.** The script
   scripts/build_finite_measurement_rank_discrimination.py partitions the 30
   seeds into disjoint 15-seed halves and compares the two empirical
   pseudo-contexts. This is a finite-sample diagnostic, not a second
   definition of the primary table.
3. **Seed-block audit.** The per-seed rows retain both measurement replicates
   and report seed-preserving cross/floor quantities and held-out stable-order
   summaries. They diagnose sensitivity to the sampling unit and are not
   pooled into the primary \(D_{\mathrm{adj}}\) claim.

Stable directions always use the registered Beta direction posterior and the
fixed minimum strict support of 8. Ties are retained as a third state and are
not silently converted to strict directions.
