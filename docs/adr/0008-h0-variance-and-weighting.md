# ADR 0008 — H0: uncentred variances, 1/τ weights, identification of the fractional part

**Status**: accepted (2026-11-08). Point (1) and the negative control amended by
[ADR 0015](0015-volume-centred-order-flow.md) (deviations D3, D4): on real data the daily
imbalance rides on clustered activity, so the flow is now centred day by day in
proportion to volume and the fit models the centring exactly. Points (2) and (3) stand.

**Context**: three problems found in simulation and property tests.

1. Subtracting the sample mean of overlapping increments biases the variance down at large
   τ under long memory (≈ −0.05 on H at H = 0.9).
2. With equal weights in log space the 4-hour points, estimated from few independent
   increments, dominate the curvature: the standard deviation of Ĥ0 doubles.
3. On Brownian data (shuffled signs) the mixed model is flat in H and the fit drifts to a
   bound (H ≈ 1).

**Decision**: (1) the principal uses uncentred mean squared increments: the model has no
drift and the daily net imbalance of the real flow is part of its long memory; (2) weighted
NLS with weights ∝ 1/τ, the inverse variance of `log V̂(τ)` under the Brownian null (the
"weighted" fit of spec §5.2); (3) the fractional component is retained only if, at the
largest τ, it explains more than twice the sampling error of the variance there
(`2·sqrt(4m/3n)`, floor 5 %); otherwise `H0 = 1/2`. The shuffled-sign control removes each
day's mean increment (the permutation keeps the day's imbalance, a drift) and corrects the
shrinkage this causes exactly.

**Consequences**: in simulation (12 days), |bias| ≤ 0.02 for H0 ∈ [0.6, 0.9] with
noise/signal up to 2, and shuffled signs give 0.49–0.52.
