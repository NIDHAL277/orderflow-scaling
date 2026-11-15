# orderflow-scaling

This project checks, pair by pair on about sixty cryptocurrencies, whether a single
parameter — the persistence `H0` of the signed order flow — fixes the roughness of volume,
the exponent of market impact and the roughness of volatility, as predicted by
Muhle-Karbe, Ouazzani Chahdi, Rosenbaum & Szymanski (2026).

| Prediction | Exponent | `H0 = 3/4` |
|---|---|---|
| Signed flow | `H0` | 0.75 |
| Volume rate | `H0 − 1/2` | 0.25 |
| Extraday volatility | `2H0 − 3/2` | 0 |
| Impact (constant-rate meta-order, in time) | `2 − 2H0` | 0.5 |

<!-- results:start -->
**Result:** *pending: the confirmatory tests run on the frozen confirmation sample.*
<!-- results:end -->

* [Architecture](architecture.md): six stages, one store, four outputs.
* [Data](data.md): Binance Vision formats and quality controls.
* [Estimators](estimators.md): what is estimated, how, and what simulation says about it.
* [Protocol](protocol.md): pre-registration, development and confirmation samples.
* [Operations](operations.md): daily run, monitoring, incident procedures.
* [Compliance](compliance.md): every requirement of the specification, with its evidence.
* [Design decisions in ten questions](design-faq.md): the choices a reviewer will question, answered.

Beyond the core tests:

* [Forecasting](forecasting.md): does roughness forecast daily volatility and volume? An
  exact rough-volatility predictor against HAR, out of sample, with live daily forecasts.
* [Change-point monitor](changepoints.md): CUSUM and Shiryaev–Roberts alarms when a pair's
  `H0` leaves its development range.
* [L2 collector](l2-collector.md): local order books of the perpetuals from the WebSocket
  stream, sampled every second with every trade.

"Production grade" here means: one command rebuilds every result from scratch, the
pipeline updates itself every day, every module has its tests, everything runs in a
container and reports its own failures.
