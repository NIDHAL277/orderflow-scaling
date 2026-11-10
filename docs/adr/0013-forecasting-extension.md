# ADR 0013 — Forecasting extension: exact rough predictor, model-based levels

**Status**: accepted (2026-11-10)

**Context**: the owner asked for a forecasting part. The specification lists forecasting
nowhere and excludes trading strategies and P&L backtests (§4). The scaling laws predict
the roughness of volatility from order flow (`H_σ = 2 H0 − 3/2`); roughness is known to
matter for volatility forecasts (Gatheral, Jaisson & Rosenbaum, 2018, §5). The 5-year daily
history of ADR 0011 gives enough days for rolling out-of-sample forecasts.

**Decision**:

* A separate family of hypotheses F1–F3 (pre-registration §10), evaluated with statistical
  losses only: QLIKE (primary) and squared error on the level of the h-day average,
  Diebold–Mariano tests pooled across pairs, model confidence set per pair. No trading rule.
* The principal rough model is the **exact Gaussian predictor** of the model the Whittle
  estimator already fits (day-averaged fBm plus white noise), parameters by Whittle on a
  500-day window. The published GJR kernel is kept as a benchmark (`rough_gjr`): in a probe
  it lost 13 % of one-step MSE to the exact predictor on daily-sampled fBm, and more with
  noise, so making it the rough model would test the discretisation, not roughness.
* Log forecasts become levels with each model's own **lognormal correction**; Duan
  smearing stays as an option. In probes, smearing of overlapping 22-day targets cost the
  exact predictor about 10 % of QLIKE against 0.5 % for the lognormal correction.
* The theory link (F2) compares H fixed at `2 Ĥ0 − 3/2` with H fixed at the development
  estimate of H_σ, everything else equal; on the confirmation sample both use only the
  development sample, so the test is out of sample. On the development sample these rows
  are flagged `look_ahead`.
* The daily history gains the traded quote volume and trade count (stage `history_rv` 3,
  before any history file existed), for the volume target F3.
* `forecast` is excluded from the checkpoint key of the simulation study: it does not
  change a simulated cell, and the full study was running when the section was added.

**Consequences**: one more command (`ofs forecast`), outputs in `data/forecast/`, three API
endpoints, a dashboard page, a report table and figure. Forecasting adds about one CPU-hour
per sample with 120 series. The probes behind the two design choices are summarised in
`docs/forecasting.md`; the reproducible check is `ofs simulate forecast`.
