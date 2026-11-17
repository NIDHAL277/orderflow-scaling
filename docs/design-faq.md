# Design decisions in twelve questions

The choices a reviewer is most likely to question, each with the short answer, the
evidence, and where it lives in the code.

## 1. Why test the theory asset by asset rather than on average?

The theory predicts *relations* between exponents (`H_vol = H0 − 1/2`, `H_σ = 2H0 − 3/2`,
`δ = 2 − 2H0`). An average over assets can match the prediction by coincidence; a relation
must also hold *across* assets whose `H0` differ. So each pair gives a point (`Ĥ0`, `Ŷ`),
and the test is on the line through 60 points: slope and intercept jointly, against the
predicted ones. Both coordinates are estimated with error, so ordinary least squares would
flatten the slope (attenuation bias); Deming regression uses the ratio of the two error
variances, taken from the bootstrap. The risk is low power if `H0` hardly varies across
pairs; the decision rule then says "undecided" rather than forcing an answer.
Code: `ofs.inference.hypotheses.cross_test`, `ofs.inference.deming`.

## 2. How is `H0` estimated, and why weights ∝ 1/τ?

The signed flow is modelled as a fractional part plus a martingale (the paper's Theorem
3.6): `Var(X_{t+τ} − X_t) = σ1² τ^{2H0} + σ2² τ`. The aggregated variance `V(τ)` is fitted by
weighted non-linear least squares over τ from 10 s to 4 h. The sampling variance of
`log V(τ)` grows roughly like τ, so weights ∝ 1/τ are the inverse-variance weights; in
simulation they halve the standard deviation compared with equal weights and remove a
spurious `H` on Brownian data. If the fractional part is not significant at 4 h, `H0` is set
to 1/2. Code: `ofs.estimators.h0`, ADR 0008.

**Why is the flow centred day by day in proportion to volume?** Because the development
year showed that otherwise a day-level imbalance reads as memory.
* **What failed.** The negative control (signs shuffled within each day) gave 0.80
  instead of 0.5. The shuffle keeps each day's net buying or selling, which rides on
  bursts of activity, and the control had removed a drift uniform in time.
* **How far it goes.** Without centring, memoryless signs with a 5 % daily imbalance read
  as `H0 ≈ 0.98` in simulation.
* **The fix.** Each 10-second window now loses its volume times the day's imbalance.
  Removing a day's net increment shrinks the variance at large τ like a bridge, and the
  fit uses that exact shape, so genuine memory is not biased.
* **The result.** The shuffled control returns 0.49–0.50, and real H0 drops by about 0.03.
* **Why simulation missed it.** The simulated trades arrived uniformly in time; real
  activity clusters.
* It was decided on development data before the confirmation year was touched, and is
  logged as deviations D3/D4 (ADR 0015).

## 3. Why reduce the trades to additive daily statistics?

Every estimator is written as `fit(sum of daily feature rows)`: sums of squared increments
per lag, counts, cross-products. Then a bootstrap replicate is just a weight vector over
days, the 999 replicates of 120 pair-markets never re-read a trade, raw files can be
deleted after reduction (the 24 months would not fit otherwise), and a block of 14 days for
the change-point monitor is one fit of 14 summed rows. Code: `ofs.reduce`,
`ofs.estimators.base.Features`, ADR 0004.

## 4. Why a synchronized block bootstrap, and why widen it by k?

Days are resampled by blocks (stationary bootstrap, mean 7 days) because one-day blocks
would destroy the long memory being measured; the *same* resampled days are applied to
every pair, because crypto pairs move together and the cross-sectional test must see that
correlation. Under long memory, the block bootstrap still underestimates the sampling
spread on one-year samples; the full simulation measured the ratio of true to bootstrap
standard deviation, and its median over the principal estimators' plausible cells widens
every interval: 1.14 at registration, 1.06 since the volume-centred `H0` (deviation D4),
whose intervals are better calibrated. Code: `ofs.inference.bootstrap`, `ofs.simulate.decisions`.

## 5. Why simulate before touching real data, and what did it change?

Every estimator is biased somewhere; the only way to know where is to run it on data with
a known answer, through the same reduction code. The full study (82 cells × 200
replications of 365 days) changed three choices: the paper's autocovariance GMM replaced
the moments for `H_vol` (moments biased up to 0.079 by noise); the quadratic-variation
ratio replaced the sign autocorrelation as control of `H0` (biased by 0.033); and it
showed that `H_σ` near 0.1 cannot be estimated within 0.02 on one year. The rules that
turned these numbers into decisions were written before the study ran. Code:
`ofs.simulate.study`, pre-registration §4.1.

## 6. Why Whittle for `H_σ`, and why is H2 "limited"?

Daily realised variance is a day *average* of volatility plus measurement error. The
Whittle estimator of Fukasawa, Takabatake & Westphal models exactly that (day-averaged fBm
plus white noise) and stays consistent near `H = 0`; the GJR moment regression ignores the
averaging and is biased upward near 0 (+0.121 in the study). Whittle still misses the 0.02
tolerance (0.030), so H2 is reported as limited by estimator reliability. The rule's
fallback kept GJR; deviation D1 made Whittle principal, before any real-data estimate
(`config/deviations.md`). The theory puts `H_σ` near 0 to 0.1, exactly where estimation is
hardest: that is a finding, not a failure.

## 7. Why "rejected / compatible / undecided" with a margin?

With enough data any tiny deviation is significant, and absence of significance is not
evidence of agreement. So a relation is *rejected* only if the Holm-corrected Wald test
rejects at 1 % **and** the mean gap exceeds 0.05 (Hurst units); *compatible* only if the
test does not reject **and** the 95 % interval of the gap lies within ±0.05 (an
equivalence test); otherwise *undecided*, with the number of days that would decide. The
same rule decides the forecasting hypotheses with a 2 % margin on the loss ratio.
Code: `ofs.inference.hypotheses.decide`.

## 8. Why a pre-registration enforced in code?

With four exponents, several estimators, windows, τ ranges and samples, there are hundreds
of ways to get a "significant" result. Writing the hypotheses, estimators and decision
rules down and tagging them (`prereg-v1`) before any real-data estimate, then freezing the
configuration (`config-frozen`) before the confirmation sample, removes that freedom. The
code refuses to estimate before the tags and checks that the configuration is unchanged; in
the container, CI bakes the tag list and the config hash into the image. Every later change
is a dated deviation. Code: `ofs.protocol`, ADR 0006.

## 9. Why an "exact" rough predictor, QLIKE, and a lognormal level correction?

The published rough-volatility forecast (Gatheral, Jaisson & Rosenbaum, 2018) assumes a
continuously observed, noise-free fBm; daily RV is a noisy day average. Conditioning a
Gaussian model of exactly that (the Whittle model) gives the optimal linear forecast; in
simulation the published kernel lost 13 % of one-step MSE to it. QLIKE and MSE on levels
rank forecasts correctly even with a noisy RV proxy (Patton, 2011), which the MSE of logs
does not. Each log model returns to levels with its own lognormal correction: the empirical
smearing factor was too noisy at 22 days and changed rankings. Result in simulation: about
1 % better than HAR at one day, 7–13 % at 22 days. Code: `ofs.forecast`, ADR 0013.

## 10. What broke on real data, and what changed because of it?

* USD-M trade ids go missing (0.1–0.3 % from mid-2025) while aggregate ids stay
  consecutive: continuity is checked on aggregate ids, trade-id gaps by fraction and run
  length (ADR 0012).
* Futures klines may open at the previous close: the OHLC check accepts either (ADR 0009).
* Binance re-publishes corrected files at the same URL: a weekly checksum recheck (ADR 0010).
* Perpetuals keep publishing flat klines after trading stops; klines outlive trades: days
  without trades are dropped and availability is counted on the trade listing.
* The USD-M WebSocket serves trades only on `/market` (the root accepts the subscription and
  sends nothing): the collector uses two connections (ADR 0014).
* Operations: a shell filter deleted derived data, a closed lid killed the jobs, a restart
  stopped them again. Hence explicit-name cleanup only, resumable jobs, and a resume script
  (incident notes in `docs/incidents/`).
* Small files: on the laptop's hard disk the daily cycle of 2026-11-15 took 11 hours, nearly
  all of it opening about 955,000 files of a few kilobytes. Closed months are now folded into
  one file per statistic, pair and month, read back and compared before the daily files go
  (ADR 0016); the bundles read from both layouts were identical on real data.
* The health check said "ok" with data 16 days old: it now reports "stale" once the last
  verified day is more than 3 days old.

## 11. What did the confirmation year show?

Every choice was frozen on 2026-11-14 (tag `config-frozen`) before the confirmation year
(2025-11-01 to 2026-10-31) was estimated.

* **Volume (H1)**: compatible on perpetuals (gap +0.001, interval −0.012 to +0.011),
  rejected on spot (−0.119).
* **Volatility (H2)**: rejected on both markets. With `H0` below 3/4 the relation predicts a
  negative `H_σ`, which no Hurst exponent can take; the daily estimates sit on the lower
  bound of their range, so the intervals are lopsided.
* **Impact (H3, exploratory)**: rejected on spot; undecided on perpetuals, where the gap is
  large (−0.384) but `H0` spans only 0.54 to 0.62 across pairs, too narrow to identify a
  slope. The control estimators disagree on the size of the gap, so H3 never counts alone.
* **Level of `H0` (H4)**: 0.700 on spot and 0.597 on perpetuals (development: 0.705, 0.590).
* **Forecasts (F1–F3)**: at 22 days the rough predictor's loss is 27 % below HAR's for
  volatility, but the interval is wide and the Holm-corrected test does not decide. Using
  the order-flow `H` instead of the measured one changes forecasts by less than 1.5 %:
  equivalent at 1 and 5 days.

Five of the six primary decisions of the development year were reproduced. The one change,
H3 on perpetuals (rejected, then undecided), comes from the narrow cross-section, not from a
change of method. Numbers: the README (generated from the result tables) and `paper/main.pdf`.

## 12. What are the limits, and what would come next?

* One main venue. Bybit and OKX are an exploratory check on five pairs over one month: on
  perpetuals they agree with Binance within 0.03 on average (`docs/venues.md`).
* `H_σ` near zero is where the estimators are least reliable (Whittle's bias 0.030 in the
  simulation study). H2's rejection rests on the sign of the predicted exponent more than on
  the precision of the estimate.
* `δ` uses a proxy of price impact built from trades; there is no order-book history (the
  L2 collector starts one).
* One year per sample, and the live track record of the forecasts has only just started.
* Next: run the daily pipeline on a server to build that live record, extend the venue
  comparison, and re-run the same frozen protocol on the following year as a second
  confirmation.
