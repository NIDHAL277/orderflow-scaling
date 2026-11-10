# Literature watch

Spec §2 asks for a search of arXiv ("order flow H0 crypto") and of the Rough Volatility
Network list before starting, and again every month because the tested theory dates from
2026. The monthly search is automated: `scripts/arxiv_watch.py` runs in the `literature`
workflow on the first day of each month and opens an issue listing the new papers. This page
keeps what matters for the project.

## 2026-11-09 — check before the pre-registration

Searches: arXiv API over the last 400 days (`scripts/arxiv_watch.py --days 400`: order flow
and crypto, roughness and crypto, order flow with impact and rough volatility, the authors of
the theory) and web searches on signed order flow and long memory in crypto trades. The
literature list of the [Rough Volatility Network](https://sites.google.com/site/roughvol/)
was not reviewed item by item (only its landing page was reached): to do by hand.

**Result: no study tests the relations of the theory on crypto, or estimates H0 asset by
asset.** The claimed contribution (spec §2) stands. Closest work:

| Paper | Relation to this project |
|---|---|
| Muhle-Karbe, Ouazzani Chahdi, Rosenbaum & Szymanski, [A unified theory of order flow, market impact, and volatility](https://arxiv.org/abs/2601.23172) (2026-01, v2 2026-02) | the tested theory; equities, H0 ≈ 3/4 on average; the authors have posted nothing newer |
| [Multifractality and sample size influence on Bitcoin volatility patterns](https://arxiv.org/abs/2511.03314) (Takaishi, 2025-11) and [Multifractality in Bitcoin Realized Volatility: Implications for Rough Volatility Modelling](https://arxiv.org/abs/2507.00575) (2025) | multifractal analyses of Bitcoin realised volatility; the second argues that multifractality is incompatible with a monofractal rough model, one reason why `H_σ` has several estimators and the linearity of ζ(q) is a diagnostic |
| [Memory, Roughness, and Information Persistence in Financial Markets](https://arxiv.org/abs/2605.24285) (2026-05) | equities: H ≈ 0.06 for log realised variance across 115 S&P 500 stocks |
| Kim & Hansen, [The Quarter-Hour Effect](https://arxiv.org/abs/2607.09426) (2026-07) | periodic algorithmic trading in crypto futures: intraday seasonality finer than the hour of the week; relevant to the deseasonalisation robustness check |
| Garcia Seuma, [Where does the criticality live?](https://arxiv.org/abs/2607.27070) (2026-07) | early-warning signals across crypto-perpetual liquidation cascades; same microstructure, different question |
| Caruso, [Pricing and Calibration of Bitcoin Inverse Options via the Rough Bergomi Model](https://arxiv.org/abs/2608.27575) (2026-08) | option-implied roughness for Bitcoin; no order-flow estimate |

Earlier crypto work measures the roughness of Bitcoin volatility
([Rough volatility of Bitcoin](https://arxiv.org/abs/1904.12346), Takaishi) or the Hurst exponent
of returns ([Some stylized facts of the Bitcoin market](https://arxiv.org/abs/1708.04532)), never the signed order flow together
with volume, volatility and impact.

Next check: the `literature` workflow, every first of the month.
