# Forecasting extension (F)

The scaling laws say how rough volatility should be, `H_σ = 2 H0 − 3/2` (paper, Remark
4.2). The forecasting extension asks the practical question: **does that roughness help
forecast volatility, and does the roughness implied by order flow forecast as well as the
roughness measured on volatility itself?** Daily traded volume gets the same treatment.

It is a statistical evaluation (losses, Diebold–Mariano tests, model confidence set). The
specification excludes trading strategies and P&L backtests (§4), and so does this
extension. The design is pre-registered (`config/preregistration.md` §10) and frozen with
the rest of `config/estimation.yaml`.

```bash
ofs forecast --sample development --jobs 4      # needs the tag prereg-v1
ofs forecast --sample confirmation --jobs 4     # needs config-frozen (confirmatory)
ofs simulate forecast --reps 10 --jobs 4        # check on simulated rough volatility
```

## Target, data, origins

* **Target**: the average of the series over the next `h` days, `y_h(t) = mean(y[t+1..t+h])`,
  `h ∈ {1, 5, 22}` (day, week, month; Corsi, 2009). Series: daily 5-minute realised variance
  (`rv_5m`) and daily traded quote volume (`quote_volume`).
* **Data**: the 5-year daily history from monthly 1-minute klines (`daily_stats/rv_history`,
  [ADR 0011](adr/0011-five-year-volatility-history.md)), cut at the end of the evaluated
  sample: no later day is read.
* **Origins**: every day of the evaluated period. `development`: 2023-05-01 → 2025-10-31
  (the first origin needs a 500-day window); `confirmation`: 2025-11-01 → 2026-10-31.
  Forecasts use the data up to and including the origin only; a test perturbs every value
  after the origin and checks that no forecast moves.

## Models

| Model | Forecast of the log series | Parameters (rolling 500-day window) |
|---|---|---|
| `rw` | last value | none |
| `ar_log` | AR(5) on log y, direct projection on `log y_h` | OLS, refitted daily |
| `har` | HAR of Corsi (2009) on levels | OLS, insanity filter (in-window range) |
| `har_log` | HAR on the logs of the daily, weekly, monthly averages (benchmark) | OLS |
| `rough_gjr` | kernel of Gatheral, Jaisson & Rosenbaum (2018, §5) as published | H by their moment regression |
| `rough_rolling` | exact predictor (below) | H, noise ratio, scale by Whittle |
| `rough_dev` | exact predictor | H = development estimate of H_σ; noise ratio by Whittle |
| `rough_theory` | exact predictor | **H = 2 Ĥ0 − 3/2** (development H0, floored at 0.01) |

Rough parameters are re-estimated every 5 days, regressions every day.

### The exact rough predictor

The daily log RV is modelled as the Whittle estimator of Fukasawa, Takabatake & Westphal
already models it for H_σ: the **day average of a fractional Brownian motion plus white
measurement noise**, `Y_t = c + ν · mean_{day t} W^H + ε_t`. Its increments are stationary
with autocovariance `ν² [g_H(k) + r (2·1{k=0} − 1{|k|=1})]`, where `r = var ε / ν²` and

```
g_H(k) = ½ δ⁴Φ(k),   Φ(x) = |x|^{2H+2} / ((2H+1)(2H+2))      (fourth central difference)
```

is the autocovariance of the increments of day-averaged fBm (a test checks it against the
spectral density the Whittle estimator uses, to 6 digits). The forecast of `Y_{t+D}` is the
Gaussian conditional expectation given the last 499 increments: one Toeplitz system per
horizon (Levinson recursion), weights on past values summing to one (no level parameter).
The same algebra gives the prediction-error variance, hence the level forecast
`E[e^{Y_{t+D}}] = exp(forecast + ½ var)`. A test checks on simulated paths that the
one-step mean squared error equals that variance.

### Why not just the published kernel

The GJR (2018) formula is the prediction of a *continuously observed, noise-free* fBm with
an infinite past. Daily RV is neither: it is a daily average and it carries estimation
noise. In a probe on simulated fBm sampled daily (H = 0.1, 200-day window), the one-step
mean squared error of the kernel discretised by cell integrals was 13 % above that of the
exact Gaussian predictor (midpoint cells: 5 %); the kernel puts 63 % of the weight on the
last day. With measurement noise the gap widens. `rough_gjr` is kept as the published
benchmark, `rough_rolling` is the principal rough model.

### From log forecasts to levels

Every log model maps its forecast to a level with its own lognormal correction
`exp(μ + s²/2)`: in-window residual variance (regressions), model-implied conditional
variance (exact predictor), `c_H K D^{2H}` with `c_H = Γ(3/2−H) / (Γ(H+1/2) Γ(2−2H))`
(GJR kernel). The empirical smearing factor of Duan (1983) is available
(`level_correction: smearing`) but is noisy at long horizons: 250 overlapping 22-day
ratios hold about 11 independent blocks. With the true parameters, smearing cost the exact
predictor about 10 % of QLIKE at 22 days in the probe, the lognormal correction 0.5 %.

## Evaluation

* **Losses** on the level of the target: QLIKE `y/f − log(y/f) − 1` (primary) and squared
  error. With a noisy but conditionally unbiased proxy (daily RV for integrated variance)
  both rank forecasts as the latent variance would (Patton, 2011); the squared error of
  logs does not and is not used.
* **Per pair**: mean losses on the dates where every model forecasts, ratio to `har_log`,
  Diebold–Mariano test against `har_log` (Newey–West with `max(h−1, ⌊4(T/100)^{2/9}⌋)` lags,
  Harvey–Leybourne–Newbold correction, Student t), model confidence set of Hansen, Lunde &
  Nason (2011) at 90 % (T_max statistic, stationary bootstrap of dates, mean block 22 days).
* **Pooled across pairs** (hypotheses): per pair the loss differential is scaled by the mean
  loss of the reference model, averaged across the pairs present each date, and the DM test
  is run on that single series (cross-sectional correlation between cryptos is then
  harmless). Its mean is the average relative change of the loss, `ratio − 1`, with a
  stationary-bootstrap 95 % CI over dates.
* **Decision rule**: the one of the main tests, on the relative loss change, with margin
  0.02: *better* / *worse* when the Holm-corrected DM test rejects at 1 % and
  `|ratio − 1| > 0.02`; *equivalent* when it does not reject and the CI lies within
  `[0.98, 1.02]`; *undecided* otherwise.

| Hypothesis | Target | Model / reference | Reading |
|---|---|---|---|
| F1 | volatility | `rough_rolling` / `har_log` | does roughness forecast better than HAR? |
| F2 | volatility | `rough_theory` / `rough_dev` | does the order-flow H forecast as well as the volatility H? The theory passes if *equivalent* |
| F3 | volume | `rough_rolling` / `har_log` | F1 for traded volume |

On the development sample `rough_dev` and `rough_theory` take H from estimates of the same
period: their rows are flagged `look_ahead` and are exploratory. On the confirmation sample
H comes from the development sample only, so F2 is a genuine out-of-sample test of the
scaling law.

## Simulation check

`ofs simulate forecast` simulates the model of the exact predictor (H ∈ {0.05, 0.1, 0.2},
noise sd ∈ {0, 0.3}, ν = 0.3, 1,500 days) and reports each model's QLIKE relative to
`har_log` (`data/simulation/forecast_study.parquet`, report table `tab:forecast-sim`).
Exploratory run (4 replications, H = 0.1, noise 0.3):

| Model | h = 1 | h = 5 | h = 22 |
|---|---|---|---|
| `rough_rolling` | 0.989 | 0.976 | 0.915 |
| `rough_dev` (true H) | 0.988 | 0.975 | 0.915 |
| `rough_gjr` | 1.030 | 1.053 | 0.981 |
| `ar_log` | 1.019 | 1.055 | 1.084 |
| `har` (levels) | 1.028 | 1.055 | 1.053 |
| `rw` | 1.40 | 2.10 | 2.88 |

Under rough volatility the exact predictor wins and its advantage grows with the horizon;
estimating H costs nothing against knowing it (Whittle underestimates H and the noise ratio
together on 500-day windows, with little effect on the forecasts). A weekly CI test
(`tests/simulation/test_forecast_sim.py`, marked slow) keeps these orderings.

## Live forecasts (production)

The daily run (`ofs daily`) also forecasts every pair at the end of the day just
completed, with every configured model, for the next 1, 5 and 22 days
(`ofs.forecast.live`). The series is the 5-year history (complete months, klines) followed
by the daily aggTrades statistics for the days it does not cover yet: the same 5-minute RV,
and the notional traded as the quote volume. Each day's forecasts are written once
(`data/forecast/live/{day}.parquet`) and never recomputed; as targets complete, the run
scores them (`live_scored.parquet`), which builds a live, genuinely out-of-sample track
record of every model. `GET /forecast/live` serves the latest forecasts (with the
annualised volatility `sqrt(365 × average daily variance)`); the dashboard shows them as a
"live outlook" with the track record. Live forecasts estimate on real data, so they start
once the protocol allows the rolling estimates (tag `prereg-v1`).

## Outputs

`data/forecast/`: `forecasts_{sample}.parquet` (one row per pair, market, target, model,
horizon and origin, with the realised target, both losses, H and the noise ratio),
`evaluation_{sample}.parquet` (per pair) and `tests_{sample}.parquet` (F1–F3). They are
served by the API (`/forecast/tests`, `/forecast/evaluation`, `/forecasts/{symbol}`,
`/forecast/live`), shown
on the dashboard page *Forecasting*, and in the report (table `tab:forecast`, figure 9).

## Limitations

* The rough predictor assumes the H of the window holds over the horizon; regime changes
  are not modelled by the forecasts (the change-point monitor of H0,
  [changepoints](changepoints.md), flags them separately).
* No weekday effect is modelled for any model (weekends trade less in crypto).
* `rough_theory` floors `2 Ĥ0 − 3/2` at 0.01: for `Ĥ0 < 0.755` the theory predicts a
  non-positive H_σ, which no fBm predictor can use; the number of floored pairs is reported.
