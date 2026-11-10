# Pre-registration — order-flow scaling laws on crypto (orderflow-scaling)

| | |
|---|---|
| Status | **DRAFT** — becomes binding when committed and tagged `prereg-v1` (spec §9, step 0) |
| Author | nidhal.V |
| Drafted | 2026-11-08 |
| Theory tested | Muhle-Karbe, Ouazzani Chahdi, Rosenbaum & Szymanski, *A unified theory of order flow, market impact, and volatility*, arXiv:2601.23172 (v2) |
| Configuration | `config/estimation.yaml` (hash stamped on every result as `config_version`) |

Before the tag `prereg-v1`, real data are used only to test ingestion and data quality. No
estimator is run on real data before that tag; the code enforces it (`ofs.protocol`).

## 1. Question

The theory says one parameter, the persistence `H0` of the signed order flow of "core"
orders, fixes three other exponents:

| Quantity | Predicted exponent | Value if `H0 = 3/4` |
|---|---|---|
| Signed flow (Hurst) | `H0` | 0.75 |
| Unsigned volume rate (Hurst) | `H0 - 1/2` | 0.25 |
| Volatility, extraday (Hurst) | `2 H0 - 3/2` | 0 |
| Impact of a constant-rate meta-order (time exponent) | `2 - 2 H0` | 0.5 |

Reading of the paper fixed before coding (Thm 3.4-3.5, Sec. 3.3, Thm 4.1, Remark 4.2):
the volume prediction concerns the volume **rate** (cumulative volume is the integral of a
rough process); the volatility prediction concerns the **extraday** volatility; the impact
exponent is an exponent in execution **time** for a constant-rate meta-order.

Question: do the relations hold **asset by asset** across crypto pairs, or only on average?

## 2. Data and samples

* Binance Vision `aggTrades`, spot and USD-M perpetuals; 1-minute klines for the OHLC check.
* Universe (`config/universe.yaml`, frozen and dated before this tag): USDT spot pairs, no
  stablecoin/stablecoin, fiat or leveraged token, ranked by median daily quote volume over
  2024-11-01 → 2026-10-31, the first 60 with a daily aggTrades file on ≥ 95 % of days; the
  USD-M perpetual of the same crypto when its aggTrades cover ≥ 95 % of days too (klines
  alone do not count: the RAYUSDT perpetual has klines but no trades in the window); renames
  merged through the versioned table `config/symbol_renames.yaml`, the days of a token swap
  being explained gaps.
* Left out by the author's decision on 2026-11-09, before any estimation: PAXGUSDT (a
  gold-backed token: its price follows gold, not crypto order flow) and WBTCUSDT (wrapped
  bitcoin: a copy of BTC's price that would count BTC twice). The next pairs by liquidity,
  ARKMUSDT and DASHUSDT, take ranks 59 and 60 (`selection.excluded` in the universe file;
  `make universe` applies the same exclusions). Universe: 60 pairs, 59 with a perpetual.
* Signs: `ε = +1` if `is_buyer_maker` is false, `-1` otherwise; flows in USD notional.
* Daily volatility history (spec §4, "up to 5 years where the history exists"): daily
  5-minute RV from the monthly 1-minute klines, 2021-11-01 → 2026-10-31 (`history`
  sample). It feeds one robustness variant of H2 only (§8); every test uses the four
  exponents estimated on the same sample.
* Windows Δ ∈ {1, 5, 10, 30, 60} s (1 s only for the 10 most liquid pairs); weekly
  seasonality removed by an hour-of-week profile (168 cells, trailing 84 days).
* **Development sample**: 2024-11-01 → 2025-10-31 (choice of τ ranges, windows, gaps).
* **Confirmation sample**: 2025-11-01 → 2026-10-31. Not estimated before the tag
  `config-frozen`; the code refuses to (`ofs.protocol`).

## 3. Hypotheses

Each relation is tested by a cross-sectional regression over the N pairs, `Ŷ_i = a + b Ĥ0_i`.

| Hypothesis | Relation | Prediction (a ; b) | Status |
|---|---|---|---|
| H1 — volume | `H_vol = H0 - 1/2` | (−0.5 ; 1) | Confirmatory |
| H2 — volatility | `H_σ = 2 H0 - 3/2` | (−1.5 ; 2) | Confirmatory |
| H3 — impact | `δ = 2 - 2 H0` | (2 ; −2) | Exploratory (proxy measure) |
| H4 — level | `H0 ∈ [0.70, 0.85]` for the 20 most liquid pairs | — | Descriptive |
| H5 — robustness | identical exponents on spot and perpetual of the same crypto | — | Robustness |

H1–H3 are run separately on spot (primary) and perpetual markets.

## 4. Estimators

A result counts only if the principal and the control estimator agree within 0.05
(per pair: flag `concordant`; per hypothesis: same decision and mean gaps within 0.05).

| Exponent | Principal | Control | Reported, not tested |
|---|---|---|---|
| `H0` | mixed fBm on aggregated variances, `Var(X_{t+τ} − X_t) = σ1² τ^{2H0} + σ2² τ`, weighted NLS (weights ∝ 1/τ), τ log-spaced from Δ = 10 s to 4 h; fractional part required to exceed twice the sampling error of `Var` at 4 h, else `H0 = 1/2` | sign autocorrelation in event time, `C(ℓ) ~ ℓ^{-γ}`, `H0 = 1 − γ/2`, ℓ ∈ [10, 10⁴], significant leading bins only | simple fBm slope (bias reproduction), DFA-2, quadratic-variation ratio on (Δ, 2Δ, 4Δ) (the paper's estimator), all windows Δ |
| `H_vol` | see §4.1 | see §4.1 | autocovariance-GMM of increments truncated at 3 sd (the paper's method), paper bins 5–12 s |
| `H_σ` | see §4.1 | see §4.1 | p-variation (Cont & Das), intraday 15-min horizon, TSRV |
| `δ` | proxy A: runs of same-sign trades < 5 s apart, `E[I | Q] = A Q^δ` on log-binned notional | proxy A on run duration (time exponent, as in the theory) | proxy A at gaps 1 s and 30 s; proxy B aggregate impact (Patzelt & Bouchaud) |

### 4.1 Decisions to settle from the simulation study, **before** tagging

The protocol (spec §5, §9 step 1) validates every estimator by simulation first. Preliminary
runs (development notes, 2026-11-08) found:

* GJR moments on daily 5-minute RV are biased upward near `H_σ = 0` (≈ +0.13 to +0.19 for
  `H_σ ∈ {0.05, 0.1}` on 365 days of rough Bergomi), because daily integration smooths the
  path; the Whittle estimator of Fukasawa, Takabatake & Westphal is much closer.
* Moments of the volume rate are biased by observation noise (≈ −0.07 at noise/signal = 1);
  the autocovariance-GMM is not.

Rule, fixed now and applied to the **full** simulation study (`make sim`), before the tag:
for each of `H_σ` and `H_vol`, the principal estimator is the spec's default (GJR daily;
moments on the rate) **if** its bias is within ±0.02 in every cell `true ∈ [0.05, 0.15]`
(H_σ) / `[0.1, 0.4]` (H_vol) with noise/signal ≤ 1; otherwise the alternative (Whittle;
autocovariance-GMM) is principal if its bias is, and the other becomes the control. If
neither is, the default is kept and the hypothesis is reported as limited by estimator
reliability. Coverage is then measured on the CIs widened by the calibration factor `k` of
§5, as they will be reported on real data; a principal estimator whose calibrated coverage
leaves [93 %, 97 %] in these cells is reported with that reliability flag. The computation
is `ofs simulate decide --profile full` (`bias_then_calibration`); `--markdown` prints the
outcome block pasted below. The outcome is written
here, in `config/estimation.yaml` (`primary_estimator`, `control_estimator`), and committed
with the tag.

*Amended on 2026-11-09, while the full study was running and before it had produced any
result, on the author's decision.* The first version also required the coverage of the raw
bootstrap CIs to lie in [93 %, 97 %]. Under long memory the day-block bootstrap
under-covers on any finite sample (quick study: 40-85 %), which is why §5 calibrates it;
judging raw coverage would have kept estimators known to be biased (GJR ≈ +0.1, volume
moments ≈ −0.07) only because no estimator can pass that test.

The same rule fixes the concordance control of `H0`: `sign_acf` (spec default) unless the
full study shows |bias| > 0.02 for it at `H0 ∈ [0.7, 0.85]`, in which case the paper's
quadratic-variation estimator `qv_ratio` is the control.

Quick-profile evidence (2026-11-09; 40 replications, 99 bootstrap draws, 20 days; 120 days
for `H_σ`), to be superseded by the full study:

| Exponent | Estimator | Bias | Coverage of the 95 % CI |
|---|---|---|---|
| `H0` | mixed fBm | within ±0.02 in all 20 cells | 40-85 % (see §5, calibration) |
| `H0` | sign ACF (LMF signs) | ≤ 0.02 for `H0 ≤ 0.7`, −0.04 to −0.07 above | 0-92 % |
| `H_vol` | moments / log-moments | > 0.02 as soon as noise > 0 | ≈ 0 % with noise |
| `H_vol` | autocovariance-GMM | within ±0.02 up to noise/signal 1 (except `H = 0.1`) | 22-90 % |
| `H_σ` | GJR, daily RV | +0.08 to +0.14 everywhere | 8-62 % |
| `H_σ` | Whittle, daily RV | RMSE 0.10-0.21 on 120 days | 72-100 % |
| `δ` | proxy A | ≈ 0 (RMSE ≤ 0.002) | not informative at this precision |

> Outcome of the rule: *to be filled in from `make sim` before `prereg-v1`.*

### 4.2 Grids (from `config/estimation.yaml`)

τ grid: 30 log-spaced values in [1 s, 4 h] plus Δ, 2Δ, 4Δ · q ∈ {0.5, 1, 1.5, 2, 3} ·
volume lags 1–360 windows (20 log-spaced) · daily volatility lags 1–50 days (12 log-spaced)
· intraday lags 1–48 slots of 15 min · proxy-A gaps {1, 5, 30} s · proxy-B horizons
{60, 300, 900, 3600} s · seed 20261106.

## 5. Cross-sectional test

* Uncertainty: stationary block bootstrap over days (Politis & Romano), **synchronized**
  across all pairs and both markets (same resampled days), mean block 7 days (sensitivity:
  3 and 14 days), B = 999 replications. Each replicate re-estimates every exponent.
* Line: Deming regression, error-variance ratio `δ = mean_i var*(Ŷ_i) / mean_i var*(Ĥ0_i)`
  from the bootstrap; OLS is reported for comparison only.
* Joint Wald test of `(a, b) = (a0, b0)` with `Σ` the bootstrap covariance of `(â, b̂)`,
  `W ~ χ²(2)`; Holm correction over H1–H3.
* Mean gap `ḡ = mean_i (Ŷ_i − a0 − b0 Ĥ0_i)` with its bootstrap 95 % CI; per-pair gaps
  with CIs (forest plot).
* Calibration of the bootstrap spread: under long memory the block bootstrap underestimates
  the sampling spread on short samples (simulated `H0 = 0.75`: true sd / bootstrap sd = 1.48
  at 20 days, 1.22 at 60, 1.13 at 120; coverage 72 %, 78 %, 85 %). Before `prereg-v1`,
  `inference.bootstrap.sd_inflation = k` is set to the median, over the cells of the
  principal estimators in the plausible region (`H0 ∈ [0.7, 0.85]`, noise/signal ≤ 1), of the
  ratio `sd_ratio` measured by the full study at 365 days (k ≥ 1). Every CI half-width is
  multiplied by k and the covariance of the Wald test by k²; k = 1 results are reported too.

## 6. Decision rule

* **Rejected** if the joint Wald test rejects at 1 % after Holm correction over H1–H3
  **and** `|ḡ| > 0.05` (Hurst units).
* **Compatible** if the test does not reject **and** the 95 % CI of `ḡ` lies within
  `[−0.05, 0.05]`.
* Otherwise **not decided** ("non tranchée"): reported as such, with the number of days
  that would be needed to decide (CI width assumed ∝ 1/√T).

A well-documented rejection is worth as much as a confirmation.

## 7. Negative controls and power (spec §9, step 3)

* Signs permuted within each day must give `H0 = 0.5` up to error (principal estimator with
  day-level centring, see `docs/estimators.md`).
* Synthetic universes violating the relation by 0.05, 0.10 and 0.20: power of the test;
  minimal effect detectable at 80 % power with N = 60; size at 1 % on synthetic universes
  satisfying the relation must lie in [0.5 %, 2 %].

## 8. Robustness (spec §9, step 4) — descriptive

Windows Δ; τ range (4 h vs 1 h); with / without deseasonalisation; spot vs perpetual;
high / low volatility regimes (cross-sectional median daily RV above / below its median);
without the 5 most liquid pairs; without days with a price jump > 20 % in one second;
block length 3 / 14 days; control estimators; `H_σ` estimated on the 5-year daily RV
history (`daily_5y`) against `Ĥ0` of the sample — its bootstrap resamples days of the
five years, so pairing the replicates treats the two estimation errors as independent
although the five years include the sample year. Shown together as a specification
curve, not corrected for multiplicity. Negative results are reported with the same care.

## 9. Deviations

Any change after this tag (data, estimator, grid, rule) goes into a section "Deviations
from protocol" of the report, with its reason and its effect on the results. The configuration
is frozen a second time by the tag `config-frozen` before the first estimation on the
confirmation sample.
