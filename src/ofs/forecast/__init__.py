"""Extension F: out-of-sample forecasting of daily volatility and traded volume.

The scaling laws say how rough volatility is (``H_sigma = 2 H0 - 3/2``). This package asks
whether that roughness forecasts: the rough predictor of Gatheral, Jaisson & Rosenbaum
against HAR-type benchmarks, with H estimated on volatility or implied by the order flow.
Statistical evaluation only (losses, Diebold-Mariano, model confidence set): no trading
rule and no P&L, which the specification excludes (§4).

* :mod:`ofs.forecast.rough`: kernel of the rough predictor;
* :mod:`ofs.forecast.models`: the competing models on one daily series;
* :mod:`ofs.forecast.evaluate`: losses and tests;
* :mod:`ofs.forecast.run`: every pair of the universe, outputs in ``data/forecast/``.
"""
