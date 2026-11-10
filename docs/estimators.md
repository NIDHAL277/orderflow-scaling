# Estimators

Every estimator is a pure function of daily sufficient statistics, split into
`features(stats, days)` (a days × K matrix of additive daily features) and
`fit(sums)` ([ADR 0004](adr/0004-additive-daily-statistics.md)). Roles (principal,
control, robustness, diagnostic, negative control) are assigned in
`ofs.estimators.registry` from `config/estimation.yaml`.

## H0 — signed flow

`X` is the cumulative deseasonalised signed notional on windows of Δ.

* **Principal — mixed fBm**: `E[(X_{t+τ} − X_t)²] = σ1² τ^{2H0} + σ2² τ`, weighted NLS on
  log-variances with weights ∝ 1/τ (inverse variance under the Brownian null), τ from Δ to
  4 h. The fractional component must explain, at 4 h, more than twice the sampling error of
  the variance there (`2·sqrt(4m/3n)`, floor 5 %); otherwise the flow is Brownian and
  `H0 = 1/2` (slope of the log-log fit) ([ADR 0008](adr/0008-h0-variance-and-weighting.md)).
* **Control — sign autocorrelation**: `C(ℓ) ~ ℓ^{−γ}` in event time, `H0 = 1 − γ/2`, on the
  leading run of lag bins significantly above zero (2 sd under no memory). Fitting all
  positive bins of a memoryless sequence returned a spurious 0.75 in testing.
* Diagnostics: simple fBm slope (reproduces the paper's bias rising with Δ), DFA-2, and the
  paper's own estimator `H = ½ log2[(V(4Δ) − 2V(2Δ)) / (V(2Δ) − 2V(Δ))]`.
* Negative control: signs permuted within each day; the permutation keeps each day's net
  imbalance (a drift, not memory), so this control removes the day's mean increment and
  corrects the induced shrinkage exactly under the Brownian null.

## H_vol — volume rate

`v_k = V_k / Δ`, deseasonalised. Principal: moments `m(q, ℓ) = mean|v_{k+ℓ} − v_k|^q ~
ℓ^{q H}` with a linearity check of `ζ(q)`. Control: the same on `log v`. Also computed: the
paper's autocovariance-GMM of increments truncated at 3 sd, which models white
observation noise explicitly.

## H_σ — volatility

The theory predicts the **extraday** roughness (paper, Remark 4.2): daily horizon primary.

* GJR moments of `log σ` increments on daily 5-minute RV (lags 1–50 days); TSRV variant.
* Whittle (Fukasawa, Takabatake & Westphal): increments of log RV modelled as increments
  of a day-averaged fBm plus white measurement noise, spectral density
  `f(λ) = ν²[4 c_H (1 − cos λ)² Σ_k |λ + 2πk|^{−2H−3} + r (1 − cos λ)/π]`; its bootstrap
  re-estimates on block-resampled series without cross-block increments.
* p-variation (Cont & Das): `H = 1/p*` where the p-variation stops depending on the step.
* Intraday horizon (15-minute local variances of 10-second returns): descriptive.
* Long history (spec §4): the same daily estimators on up to 5 years of daily RV rebuilt
  from monthly 1-minute klines (`ofs history`, sample `history`, variant `daily_5y`);
  a robustness variant of H2 ([ADR 0011](adr/0011-five-year-volatility-history.md)).

## δ — impact

Proxy A: runs of same-sign trades less than g seconds apart, `I = ε (log p_last − log
p_first)`, fit `E[I | Q] = A Q^δ` on log-binned notional (control: on run duration).
Proxy B: aggregate impact against the normalised imbalance (descriptive).

## What the simulations say

Quick profile (`ofs simulate study --profile quick`, 2026-11-09: 40 replications per cell,
99 bootstrap draws, 20 simulated days, 120 for `H_σ`):

| Exponent | Finding | Consequence |
|---|---|---|
| H0 | mixed fBm (1/τ weights): \|bias\| ≤ 0.02 in all 20 cells (H0 0.6-0.9, noise/signal 0-2) | principal kept |
| H0 | equal weights in log space: same bias, twice the sd; spurious H on Brownian data | weights 1/τ ([ADR 0008](adr/0008-h0-variance-and-weighting.md)) |
| H0 | sign ACF on Lillo-Mike-Farmer signs: biased low for H0 ≥ 0.75 (finite-size effect of the model) | control choice ruled in the pre-registration |
| H_vol | moments and log-moments biased by observation noise; autocovariance-GMM unbiased up to noise/signal 1 | rule in the pre-registration §4.1 |
| H_σ | GJR on daily RV biased up by ≈ 0.1 everywhere; Whittle much closer on 365 days | rule in the pre-registration §4.1 |
| δ | proxy A unbiased | — |
| all | 95 % bootstrap CIs too narrow on short samples: H0 coverage 72 % (20 days), 78 % (60), 85 % (120) | pre-registered calibration `sd_inflation` from the full study |

The coverage problem is the known weakness of block bootstraps under long memory: the
bootstrap spread approaches the true one as the sample grows (true sd / bootstrap sd 1.48
at 20 days, 1.13 at 120), and longer blocks do not help (1.38 with 30-day blocks at 120
days: fewer blocks make the bootstrap itself noisier). The `full` profile therefore
simulates 365-day samples, like the real ones, and reports `sd_ratio` per cell.

The definitive tables come from `make sim` (profile `full`) and are part of the report.
