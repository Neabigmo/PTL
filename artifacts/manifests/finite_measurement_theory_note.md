# PTL finite-measurement theory and literature note

**Status.** Executed theory/provenance note for the PTL scientific upgrade.

**Audit date.** 2026-09-14 (Asia/Shanghai).

**Scope.** This note records checks against the existing frozen rank-comparator,
matched-fixed measurement, ordering-identifiability, and decomposition artifacts.
It does not create a new contract, baseline, hash, gate, or manuscript edit.
The only outputs of this task are this note and
artifacts/manifests/finite_measurement_theory_note.json.

## 1. Frozen estimand and notation

For metric \(m\), context \(e\), and an unordered perturbation pair
\(a=(p,q)\), let \(R_{p,e}^{(r)}\) be the realized risk in measurement
replicate \(r\). The frozen pair state is

\[
Z_{a,e}^{(r)}=\operatorname{sign}\!\left(R_{p,e}^{(r)}-R_{q,e}^{(r)}\right)
\in\{-1,0,+1\},
\]

with \(0\) retained as a first-class tie. Its pair law is

\[
\pi_{a,e}=\bigl(\Pr(Z=-1),\Pr(Z=0),\Pr(Z=+1)\bigr).
\]

The primary transport estimand is the measurement-adjusted pair-order
discrepancy, averaged over the declared matched pair universe,

\[
D_{\rm adj}
=D_{\rm cross}-\frac{D_{\rm within,s}+D_{\rm within,t}}{2}
=\frac12\,\mathbb E_a\left\|\pi_{a,s}-\pi_{a,t}\right\|_2^2.
\]

The canonical finite-measurement value uses the existing per-seed
U-corrected within-context terms. The cross-fit stable-inversion fraction is a
separate held-out diagnostic: it selects stable pair states with the existing
Beta posterior and minimum strict support of eight, then evaluates those
states on disjoint seed halves. Continuous rank distances and top-\(k\)
comparators remain secondary diagnostics.

## 2. Frozen-artifact verification

The following checks were recomputed from the existing CSV/JSON files rather
than from manuscript text.

| Check | Existing source | Result | Disposition |
|---|---|---:|---|
| Deterministic no-tie Kendall relation | artifacts/manifests/rank_comparator_benchmark.csv and .json | 18 full-depth rows; 12 rows have no ties in either vector; maximum absolute error \(8.326672684688674\times10^{-17}\) | Verified only on the 12 tie-free rows |
| Canonical \(D_{\rm adj}\) reconstruction | artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_summary.csv | 18 full-depth directed rows; maximum error between cross_disagreement - measurement_floor and identifiable_divergence is \(8.326672684688674\times10^{-17}\) | Exact scalar reconstruction up to floating-point roundoff |
| Tie/directional cross-fit report | artifacts/manifests/reliability_transport_decomposition.csv and .json | 18 rows; 12 have zero cross-fit tie component and 6 have a positive tie component; maximum absolute cross-fit-total minus canonical \(D_{\rm adj}\) is \(0.022131164633510586\) | Estimator comparison, not an identity check |
| Known pair-law MMD check | artifacts/manifests/ordering_identifiability_synthetic.csv and source-defined laws | \(\pi_s=(.70,.20,.10)\), \(\pi_t=(.65,.30,.05)\) give \(D_{\rm adj}=0.0075\) exactly | Confirms the delta-kernel algebra |
| Finite-replicate correction | artifacts/manifests/ordering_identifiability_synthetic.csv | Under the null, plug-in bias is \(0.04658610026041667\), while U-corrected bias is \(-7.619160353536164\times10^{-5}\); under the known alternative, the corresponding biases are \(0.039060546875\) and \(-0.00037001854482321926\) | Keep the U-corrected floor; do not use the plug-in quadratic as the canonical estimate |
| Cross-fit synthetic validation | artifacts/manifests/transport_decomposition_synthetic_validation.csv | Six regimes, 300 trials, 256 pair rows; the largest absolute reported mean bias is \(1.3091001157397586\times10^{-4}\) | Supports the tie-plus-direction cross-product estimator |

### 2.1 Deterministic tie-free correspondence

For two finite vectors \(x,y\in\mathbb R^n\), define the strict inversion
fraction

\[
K_{\rm inv}(x,y)=\frac{1}{\binom n2}
\sum_{i<j}\mathbf 1\!\left\{(x_i-x_j)(y_i-y_j)<0\right\}.
\]

If neither vector has a tied item value, every unordered pair is either
concordant or discordant. Kendall's \(\tau\) is then the normalized
concordance-minus-discordance count, so

\[
\frac{1-\tau}{2}=K_{\rm inv}(x,y).
\]

The frozen comparator uses SciPy Kendall \(\tau_b\); on these tie-free rows
\(\tau_b\) has the same value as the no-tie coefficient. The artifact check
found the relation error above for all 12 applicable rows, including the
9,591-pair and 17,205-pair universes. When either deterministic vector has
ties, the no-tie inversion identity is not asserted: the frozen field
kendall_no_tie_distance is unavailable and the tie-aware pair law remains
the relevant object.

This is a correspondence limit, not a replacement for the finite-measurement
estimand. In particular, a deterministic mean vector collapses repeated
measurement uncertainty and cannot supply the within-context U-statistic
floor.

### 2.2 \(D_{\rm adj}\) as a delta-kernel MMD

Let \(S=\{-1,0,+1\}\) and define the delta kernel

\[
k_\delta(z,z')=\mathbf 1\{z=z'\},\qquad z,z'\in S.
\]

For independent \(X,X'\sim\pi_s\) and \(Y,Y'\sim\pi_t\),

\[
\begin{aligned}
\operatorname{MMD}_{\delta}^2(\pi_s,\pi_t)
&=\mathbb E k_\delta(X,X')+\mathbb E k_\delta(Y,Y')
  -2\mathbb E k_\delta(X,Y)\\
&=\sum_{z\in S}\bigl(\pi_s(z)-\pi_t(z)\bigr)^2
=\left\|\pi_s-\pi_t\right\|_2^2.
\end{aligned}
\]

The cross-context disagreement and within-context disagreement are,
respectively,

\[
d_{\rm cross}=1-\pi_s^{\mathsf T}\pi_t,
\qquad
d_{\rm within,c}=1-\left\|\pi_c\right\|_2^2.
\]

Therefore, pair by pair,

\[
d_{\rm cross}-\frac{d_{\rm within,s}+d_{\rm within,t}}2
=\frac12\left\|\pi_s-\pi_t\right\|_2^2
=\frac12\operatorname{MMD}_{\delta}^2(\pi_s,\pi_t).
\]

Averaging this equality over the fixed pair universe gives the stated
population interpretation of \(D_{\rm adj}\). The exact known-law artifact
check makes the arithmetic concrete:

\[
\begin{array}{c|c}
\text{quantity} & \text{value}\\\hline
1-\pi_s^{\mathsf T}\pi_t & 0.4800000000000001\\
1-\|\pi_s\|_2^2 & 0.4600000000000001\\
1-\|\pi_t\|_2^2 & 0.485\\
\tfrac12\|\pi_s-\pi_t\|_2^2
=\tfrac12\operatorname{MMD}_{\delta}^2 & 0.0074999999999999945
\end{array}
\]

Thus \(D_{\rm adj}\) is a half squared MMD for a categorical pair-state
law under the specified delta kernel. It is not a generic continuous MMD
between expression distributions, and it is not a claim that MMD should
replace Kendall, top-\(k\), or effect-scale diagnostics. Because the quantity
is squared, it is a discrepancy rather than a distance for the purposes of
the paper's terminology; no triangle inequality is claimed.

## 3. Exact decomposition and the order-2 U-stat attempt

For one pair, write

\[
t_c=\pi_c(0),\qquad b_c=\pi_c(+1)-\pi_c(-1)=\mathbb E_{\pi_c}[Z].
\]

The exact population algebra is

\[
\begin{aligned}
d_{\rm adj}(a)&=d_{\rm tie}(a)+d_{\rm strength}(a)+d_{\rm flip}(a),\\
d_{\rm tie}(a)&=\tfrac34(t_s-t_t)^2,\\
d_{\rm strength}(a)&=\tfrac14\bigl(|b_s|-|b_t|\bigr)^2,\\
d_{\rm flip}(a)&=|b_sb_t|\,\mathbf 1\{b_sb_t<0\}.
\end{aligned}
\]

The last two terms sum to the directional-bias quadratic

\[
d_{\rm dir}(a)=\tfrac14(b_s-b_t)^2.
\]

This identity follows by considering equal and opposite signs of \(b_s\) and
\(b_t\). It is an identity for population pair laws, not an assertion that
the three displayed population terms have unbiased finite-sample plug-in
estimates.

### 3.1 What an exact order-2 construction can estimate

Let \(f_0(z)=\mathbf 1\{z=0\}\) and \(f_1(z)=z\). If \(W_i=(X_i,Y_i)\)
denotes an independent seed-level block containing one source and one target
pair-state observation, define

\[
\delta_{f,i}=f(X_i)-f(Y_i),\qquad
U_f=\frac{1}{N(N-1)}\sum_{i\ne j}\delta_{f,i}\delta_{f,j}.
\]

Independence across the \(N\) blocks gives

\[
\mathbb E[U_f]=\bigl(\mathbb E f(X)-\mathbb E f(Y)\bigr)^2.
\]

Consequently, the exact block-level order-2 constructions for the two
quadratic components are

\[
U_{\rm tie}=\tfrac34U_{f_0},
\qquad
U_{\rm dir}=\tfrac14U_{f_1},
\qquad
U_{\rm tie+dir}=U_{\rm tie}+U_{\rm dir}.
\]

If source and target observations are unpaired independent samples, the same
quantity can be written as the two-sample U-statistic

\[
U_f^{(2s)}=
\frac{1}{n_s(n_s-1)}\sum_{i\ne i'}f(X_i)f(X_{i'})
+\frac{1}{n_t(n_t-1)}\sum_{j\ne j'}f(Y_j)f(Y_{j'})
-\frac{2}{n_sn_t}\sum_{i,j}f(X_i)f(Y_j).
\]

The existing cross-fit implementation is the contract-preserving analogue:
for independent halves \(A,B\), it uses

\[
U_f^{\rm CF}=\overline{\delta_f}^{\,(A)}
\overline{\delta_f}^{\,(B)},
\]

then applies coefficients \(3/4\) and \(1/4\). The source code implements
these terms as d_tie_cf and d_direction_cf in
src/evaluation/transport_decomposition.py. The retained two measurement
replicates are grouped inside the prescribed 15-seed halves. This preserves
the frozen seed-level separation and avoids treating all pair rows as
independent bootstrap observations.

The canonical \(D_{\rm adj}\) uses the related existing within-context
order-2 coincidence correction,

\[
\widehat W_{U,c}=\frac{1}{M_c(M_c-1)}
\sum_{r\ne r'}\mathbf 1\{Z_{c,r}=Z_{c,r'}\},
\qquad
\widehat D_{\rm within,c}=1-\widehat W_{U,c}.
\]

This is why a plug-in squared distance is not substituted for the canonical
finite-measurement value.

### 3.2 Why the full strength/flip split is not an unbiased order-2 report

Although \(d_{\rm tie}\) and the combined \(d_{\rm dir}\) are quadratic
functionals and admit the constructions above, \(d_{\rm strength}\) and
\(d_{\rm flip}\) depend on the absolute values and signs of the population
means \(b_s,b_t\). A single finite-order order-2 kernel has an expectation
that is a degree-two polynomial in the underlying law; it cannot represent
the piecewise absolute-value and sign-boundary functional for all pair laws.
The plug-in version is therefore biased near \(b_s=0\) or \(b_t=0\), and its
sign decision would also reuse the observations used for evaluation.

The frozen contract handles this boundary with a separate stable-state
posterior, a minimum of eight strict observations, and held-out cross-fit
evaluation. It does not specify an unbiased empirical strength/flip split.
Accordingly, the empirical report remains

\[
D_{\rm tie}^{\rm CF}+D_{\rm dir}^{\rm CF},
\]

while \(D_{\rm strength}+D_{\rm flip}\) is retained as population algebra
only. The observed maximum cross-fit-total versus canonical-\(D_{\rm adj}\)
difference, \(0.022131164633510586\), is estimator variation under these
distinct finite-measurement summaries, not a failed population identity.
Replacing the frozen cross-fit output with an all-row U-statistic would also
change the declared independent unit and the output contract, so it was not
implemented.

## 4. Conservative literature positioning

The intended claim is narrow: PTL supplies a tie-aware, measurement-adjusted
transport estimand for the ordering of a frozen predictor's realized risks.
The cited ranking and two-sample literatures provide neighboring concepts, not
a reason to relabel the primary estimand as a new generic ranking statistic.

| Work and existing bibliography key | Safe connection to PTL | Boundary to preserve |
|---|---|---|
| Kendall (1938), kendall1938new, *A New Measure of Rank Correlation*, *Biometrika* 30(1–2):81–93. [DOI](https://doi.org/10.1093/biomet/30.1-2.81) | Classical pairwise rank association; the frozen \((1-\tau)/2\) check is its deterministic tie-free inversion correspondence. | State the no-tie condition and \(\tau_b\) variant. Do not present \(D_{\rm adj}\) as Kendall's \(\tau\) with an informal correction. |
| Fagin, Kumar and Sivakumar (2003), fagin2003comparing, *Comparing Top k Lists*, *SIAM Journal on Discrete Mathematics* 17(1):134–160. [DOI](https://doi.org/10.1137/S0895480102412856) | Gives the top-\(k\) list-comparison context for the fixed top-10% retention/Jaccard comparator and for decision-facing shortlist summaries. | Top-\(k\) distance is a descriptive comparator here; it is not the all-pair pair-law estimand and does not remove measurement floors. |
| Vigna (2015), vigna2015weighted, *A Weighted Correlation Index for Rankings with Ties*, WWW 2015:1166–1176. [DOI](https://doi.org/10.1145/2736277.2741088) | Establishes a relevant weighted/tie-aware Kendall line when high-ranked disagreements should receive different importance. | The frozen PTL comparator is unweighted over the declared pair universe. No Vigna weight function is introduced, estimated, or used to alter \(D_{\rm adj}\). |
| Zuk, Ein-Dor and Domany (2007), zuk2007ranking, *Ranking under uncertainty*, Proceedings of the 23rd Conference on Uncertainty in Artificial Intelligence:466–473. | Provides the uncertainty-over-ranking backdrop for treating ordering as uncertain rather than forcing a single strict list. | The local entry is the citation record used here; the PTL pair law is generated by repeated realized-risk measurements and is not claimed to reproduce Zuk et al.'s ranking model. |
| Sankaran, Karlsson and Bientinesi (2025), sankaran2025ranking, *Ranking with ties based on noisy performance data*, *International Journal of Data Science and Analytics* 20:4363–4384. [DOI](https://doi.org/10.1007/s41060-025-00722-1) | Directly motivates checking how noisy repeated performance and ties affect ranking interpretation. | PTL's \(10^{-8}\) tie rule, Beta stability threshold, held-out evaluation, and U-corrected measurement floor are its own frozen procedures; equivalence to another noisy-ranking construction is not claimed. |
| Gretton et al. (2012), gretton2012kernel, *A Kernel Two-Sample Test*, *Journal of Machine Learning Research* 13:723–773. [JMLR record](https://jmlr.org/papers/v13/gretton12a.html) | Supplies the MMD framework; the delta-kernel expansion makes the categorical pair-law identity exact. | Say “half the squared delta-kernel MMD.” Do not imply a generic expression-space MMD test, a continuous-kernel result, or a new two-sample testing contribution. |

All six records are already present in manuscript/references.bib; no
bibliography or manuscript file was edited. Before any future manuscript use,
the citation key, metadata, and the boundary statement in the final column
should be checked together. In particular, the no-tie Kendall statement and
the squared-versus-unsquared MMD notation should not be shortened away.

## 5. Final disposition

The validated cross-fit decomposition is retained as the finite-measurement
report. The exact order-2 result is recorded only for the quadratic tie and
combined directional-bias functionals, under an independent seed-block
interpretation. The nonlinear strength/flip terms remain a population
interpretation, not an unbiased empirical split. No changes were made to
main.tex, supplement.tex, figure scripts, the bibliography, or any other
agent-owned file.
