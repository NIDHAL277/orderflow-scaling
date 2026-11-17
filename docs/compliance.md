# Compliance with the specification

Audit of 2026-11-09 against the *Cahier des charges — Projet 1* (28 pages, 2026-11-06):
every requirement, where it is implemented and how it is verified. **Done** means
implemented and tested; **running** means a long job is in progress; **owner** means a
decision or an action of the project owner is needed.

## Scope, data and objectives (§2-§4, §6)

| Requirement | State | Where / evidence |
|---|---|---|
| Novelty check: arXiv "order flow H0 crypto", Rough Volatility Network, every month | done | [literature watch](literature-watch.md); `scripts/arxiv_watch.py`, workflow `literature` (monthly issue) |
| Binance Vision aggTrades spot + USD-M, free, no key | done | [data sources](data-sources.md) (survey of alternatives), `ofs.ingest` |
| 60 pairs, 24 months, ≥ 95 % days, perpetual when it exists, renames | done | `config/universe.yaml` (60 pairs, 59 perpetuals, availability on aggTrades), `config/symbol_renames.yaml` (dates checked on the S3 listing) |
| Up to 5 years for daily volatility where the history exists | done (data queued) | `ofs history`, sample `history` ([ADR 0011](adr/0011-five-year-volatility-history.md)) |
| Universe frozen and dated before results | done | frozen on 2026-11-10 with the tag `prereg-v1` (content built 2026-11-09; exclusions of PAXG and WBTC decided by the owner and recorded in the pre-registration §2) |
| §6.4 blocking controls: SHA-256, ids, prices, OHLC 0.01 %, jumps, Binance corrections | done | `ofs.validate.checks`; OHLC robust to feed conventions ([ADR 0009](adr/0009-ohlc-check-in-ticks.md)); continuity on missing rows ([ADR 0012](adr/0012-trade-id-continuity.md)); corrections: `ofs recheck` ([ADR 0010](adr/0010-catch-up-and-corrections.md)) |
| §6.5 Parquet partitioned by pair and month; sizes measured | done | per-day files in `symbol=/market=/month=` partitions, folded into one pack per month once the month is closed ([ADR 0016](adr/0016-monthly-packs-of-daily-statistics.md)); about 60 MB a day measured ([operations](operations.md)) |
| O1: 60 pairs × 24 months, checksums, zero unexplained missing day | running | backfill in progress; `ofs quality` / `make quality`, dashboard health page |
| O2: bias < 0.02, coverage 93-97 % on simulated data | done, with documented failures | full study (82 cells, 200 replications of 365 days): principal estimators within 0.02 except `H_σ` (GJR +0.121, Whittle 0.030 near H = 0.05-0.15: H2 limited); coverage after calibration k = 1.14 in [80 %, 96 %] for `H0`; after deviation D4 (volume-centred `H0`, H0 cells re-run) k = 1.06 and `H0` coverage 91.5-98 %; failing cells shown in the report (pre-registration §4.1) |
| O3: four exponents with bootstrap CIs per pair | after the tag | `ofs estimate` (synchronized bootstrap, B = 999) |
| O4: tests H1-H3 reported, rejections included | after the tags | `ofs test` (Deming, Wald, Holm, decision rule) |
| O5: daily pipeline, container, tests, CI, online dashboard | done (deployment: owner) | `ofs daily`, Dockerfile, compose, CI; a VM and a public URL need the owner |

## Architecture and modules (§7-§8)

| Module | Acceptance criterion | State / test |
|---|---|---|
| M1 ingestion | re-run downloads nothing; a 404 before listing is `not_listed` | done — `tests/unit/test_download.py` |
| M2 validation | each control tested with a corrupted file | done — `tests/unit/test_validate.py` |
| M3 reduction | statistics = naive computation to 1e-9 on 3 random days | done — 3.5e-15 on real days (`ofs verify-naive`) |
| M4 simulation | fBm autocovariance within 2 %, Hawkes branching within 0.01 | done — `tests/simulation/`; fGn exact at H = 0.9 on 3.15 M points |
| M5 estimators | pure `estimate(stats, config) -> Estimate(value, diagnostics)`; diagnostics (fit, residuals, ζ(q) linearity, τ range) | done — `ofs.estimators.base`, `tests/unit/test_estimators.py` |
| M6 inference | size of the 1 % test in [0.5 %, 2 %] on synthetic universes | done — 0.88-1.32 % over 2 000 universes |
| M7 report | 7 figures and the tables in `paper/` | done — `make report`; a 10-slide deck from the same figures |
| M8 API | `/health`, `/assets`, `/estimates/{symbol}`, `/tests`, OpenAPI | done — `tests/unit/test_api_store.py`; plus `/forecast/tests`, `/forecast/evaluation`, `/forecasts/{symbol}` |
| M9 dashboard | 4 pages: tests, asset sheet, health, simulation | done — `tests/unit/test_dashboard.py`; plus a *Forecasting* page |
| M10 CLI | `ofs ingest/reduce/estimate/test/report/daily`, non-zero exit on failure | done — `tests/unit/test_cli_daily.py` |

## Protocol (§9)

| Step | State |
|---|---|
| 0. Pre-registration committed and tagged `prereg-v1` before any estimator on real data | done — tag `prereg-v1` (2026-11-10), after the full simulation study; the code refused any estimate before it |
| 1. Simulation study with failure zones in the report | done — full study, rule of §4.1 applied (`H_σ` limited); failure zones in the report |
| 2. Development then confirmation, `config-frozen` | after step 0 — enforced by `ofs.protocol` |
| 3. Shuffled signs, synthetic false universes, minimal detectable effect | done — `ofs.inference.power` |
| 4. Robustness as a specification curve | done — `ofs.inference.robustness`, 5-year volatility variant included |
| 5. Holm on H1-H3, robustness descriptive | done |
| Deviations after the tag, with reasons | D1 (Whittle principal for `H_σ`) and D2 (concordance only against unbiased controls), decided before any estimate on real data — `config/deviations.md`, paper section "Deviations from protocol" |

## Non-functional requirements and CI (§10)

| Requirement | State / evidence |
|---|---|
| Python 3.12, uv lock, `uv sync --frozen` in CI | done |
| ruff, mypy --strict | done — clean |
| Coverage ≥ 85 % (excluding `report/`) | done — 94 % (180 tests) |
| Property tests (quantity scale, shuffled signs) | done — `tests/property/` |
| One BTCUSDT day < 60 s on 4 cores, < 8 GB | done — 13.9 s, 1.4 GB on the 2025-10-10 crash (4.3 M trades, `scripts/bench_real_day.py`); `pytest-benchmark` test on 1.5 M trades |
| Backfill 24 months × 60 pairs < 24 h | running — bound by the connection (3.1 MB/s here); measured at the end of the run |
| Seeds in `config/`, PCG64, git hash and config version on every result | done (git hash once the repository has commits: owner) |
| structlog JSON, one event per stage, pair, day | done |
| NumPy docstrings, MkDocs, ADRs | done — ruff `D` rules with the NumPy convention enforced in CI |
| No secrets; pip-audit and Dependabot; API read-only with rate limit | done |
| Code MIT; raw files never redistributed; Binance terms read | done — [DATA_LICENSE](https://github.com/NIDHAL277/orderflow-scaling/blob/main/DATA_LICENSE.md): derived data CC BY-NC-SA 4.0 with attribution |
| CI: push (lint, types, tests, small simulation, Docker), weekly (full simulation, pip-audit, report), tag (image, docs) | done — `.github/workflows/ci.yml` |

## Production and deliverables (§11-§12)

| Item | State |
|---|---|
| Compose: pipeline (supercronic), API and dashboard behind Caddy, `data/` read-only for them | done |
| Daily run: 02:30 UTC, hourly retries to 12:00, rolling estimates, Healthchecks, weekly R2 backup | done — plus replay of late files and weekly application of re-published files |
| Incident procedures and three-line incident notes | done — [incidents](incidents/README.md) |
| Public GitHub repository | owner — no commit yet |
| Protocol tag, simulation tables, statistics base, H1-H5 results | in progress (see above) |
| Pipeline in production on a VM, public URL | owner — Oracle Cloud account and domain |
| Report 15-25 pages for arXiv (q-fin.TR) / SSRN | skeleton and generated figures; text of results after the tests |
| README: one-sentence result, key figure, reproduction command | reproduction done; result after the tests |
| Presentation: 10 slides, 10 minutes | done — `paper/slides.tex`, built by `make report` |

## Beyond the specification

| Item | State |
|---|---|
| Forecasting extension (owner's request): daily volatility and volume, 1/5/22 days, rough predictors against HAR, F1-F3 pre-registered (§10) | done — `ofs.forecast`, `ofs forecast`, `ofs simulate forecast`, [forecasting](forecasting.md), [ADR 0013](adr/0013-forecasting-extension.md); no trading rule or P&L, which §4 excludes |
| Extension "change-point detection on H" (§4, after v1) | done — `ofs changepoints`, weekly in the daily run, asset page and `GET /monitor/changepoints` ([monitoring](changepoints.md)) |
| Extension "L2 data of the perpetuals: continuous WebSocket collector" (§4, after v1) | done — `ofs collect-l2`, compose profile `l2` ([L2 collector](l2-collector.md), [ADR 0014](adr/0014-l2-collector.md)); the propagator model itself is future work |
| Extensions kdb+/q layer, other exchanges (§4, after v1) | Bybit and OKX: exploratory comparison of `H0` on their public trades (`ofs venues`, [Other venues](venues.md)); kdb+ not started, it needs the owner's KDB-X licence |
