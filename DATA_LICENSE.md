# Data licence

The **code** of this repository is under the MIT licence (`LICENSE`).

The **market data** come from [Binance Vision](https://data.binance.vision), under the
Binance Vision Dataset Terms (version 1.0, last updated 2026-08-26), which license the
datasets under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
licence ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) and extend
to data derived from them (charts, models, analytical indicators).

Accordingly:

* Raw Binance files are never redistributed: `data/` is not versioned, and the pipeline
  deletes each raw file once it is reduced.
* Everything this project derives from the data and makes public — daily statistics,
  estimated exponents, test results, figures and tables of the report, the dashboard and the
  API responses — is licensed under **CC BY-NC-SA 4.0**, with the attribution
  "Source: Binance Vision (data.binance.vision), CC BY-NC-SA 4.0".
* The use is non-commercial academic research (terms §4.1). No trading, signal or
  commercial product is built on the data (terms §4.2-4.4).
* This project is not affiliated with, sponsored or endorsed by Binance (terms §8.2).
  Nothing in it is financial or investment advice (terms §6).
* Downloads respect the published rate limits: at most 8 parallel requests, exponential
  back-off on errors, nothing downloaded twice (terms §7).
