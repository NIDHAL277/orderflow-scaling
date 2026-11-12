# Estimators

Every estimator is a pure function of daily sufficient statistics, split into
`features(stats, days)` (a days × K matrix of additive daily features) and
`fit(sums)` ([ADR 0004](adr/0004-additive-daily-statistics.md)). Roles (principal,
control, robustness, diagnostic, negative control) are assigned in
`ofs.estimators.registry` from `config/estimation.yaml`.

## H0 — signed flow

`X` is the cumulative deseasonalised signed notional on windows of Δ, **centred day by day
in proportion to volume** ([ADR 0015](adr/0015-volume-centred-order-flow.md), deviation
D4). Each window loses `m_d V`, its volume times the day's imbalance
`m_d = net flow / volume`. A day-level imbalance riding on bursts of activity is then not
read as memory; the uncentred flow reads a memoryless 5 % daily imbalance as H0 ≈ 0.98.

* **Principal — mixed fBm** (`mixed_fbm_vc`): `E[(X_{t+τ} − X_t)²] = σ1² τ^{2H0} + σ2² τ`.
  * Fitted by weighted NLS on log-variances, with weights ∝ 1/τ (inverse variance under
    the Brownian null), for τ from Δ to 4 h.
  * Both terms take their exact expected shape after the daily centring, a bridge:
    `g_H(m) = m^{2H} − 2(m/N)A(m) + (m/N)² N^{2H}`.
  * The fractional component must explain, at 4 h, more than twice the sampling error of
    the variance there (`2·sqrt(4m/3n)`, floor 5 %). Otherwise the flow is Brownian and
    `H0 = 1/2` ([ADR 0008](adr/0008-h0-variance-and-weighting.md)).
* **Control — quadratic-variation ratio** on (Δ, 2Δ, 4Δ), the paper's own estimator
  `H = ½ log2[(V(4Δ) − 2V(2Δ)) / (V(2Δ) − 2V(Δ))]`, on the same flow. The simulation
  study chose it over the sign autocorrelation, which was biased by 0.033.
* **Robustness:**
  * the other windows Δ;
  * no deseasonalisation;
  * τ up to 1 h;
  * `uncentred`, the pre-registered principal on the uncentred flow.
* **Diagnostics:**
  * the simple fBm slope, which reproduces the paper's bias rising with Δ;
  * DFA-2;
  * the event-time sign autocorrelation, `C(ℓ) ~ ℓ^{−γ}`, `H0 = 1 − γ/2`, fitted on the
    leading run of lag bins significantly above zero.
* **Negative control:** signs permuted within each day, then the same centring and fit;
  it must give 1/2 (deviation D3). The shuffled sign autocorrelation is no control: its
  flat autocorrelation gives `H0 = 1` by construction.

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

### Full profile (2026-11-10): the pre-registered decisions

82 cells, 200 replications of 365 days, 199 bootstrap draws (`ofs simulate decide --profile
full`; worst |bias| over the cells of the plausible region, noise/signal ≤ 1):

| Exponent | Candidates (worst \|bias\|) | Decision |
|---|---|---|
| H0 | mixed fBm 0.001 | principal |
| H0 control | sign ACF 0.033, quadratic-variation ratio 0.000 | QV ratio replaces the sign ACF |
| H_vol | autocovariance GMM 0.014, moments 0.079 | GMM principal, moments control |
| H_σ | GJR 0.121, Whittle 0.030 (H_σ ∈ [0.05, 0.15]) | neither within 0.02: the rule kept GJR; deviation D1 makes Whittle principal; **H2 reported as limited** |
| δ | proxy A 0.000 | principal |
| calibration | median true sd / bootstrap sd over 39 cells | k = 1.14 (H0 coverage 72-94 % raw, 80-96 % with k) |

Controls that fail the bias rule (moments for H_vol, GJR for H_σ) are reported but do not
decide whether a result counts (deviation D2, [config/deviations.md](https://github.com/nidhal-v/orderflow-scaling/blob/main/config/deviations.md)).

The H_σ result is itself a finding: on one year of daily data, roughness near `H = 0.1`
cannot be estimated within 0.02 by either estimator; the five-year history variant
(`daily_5y`) of the specification curve shows how much a longer sample helps.

The definitive tables come from `make sim` (profile `full`) and are part of the report.
