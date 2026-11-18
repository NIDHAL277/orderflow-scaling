# Protocol

The protocol prevents "finding" what one looks for: hypotheses frozen before the real data,
a confirmation sample never touched while tuning, every variant published.

| Step | What | Gate |
|---|---|---|
| 0 | `config/preregistration.md` committed (H1–H5, rule, estimators, grids, universe) | tag `prereg-v1` |
| 1 | simulation study of every estimator: bias, RMSE, coverage | O2, before step 0's tag |
| 2 | development sample (2024-11 → 2025-10): τ ranges, windows, gaps | tag `config-frozen` |
| 3 | negative controls and power | — |
| 4 | robustness: specification curve | — |
| 5 | Holm over the three confirmatory tests; robustness descriptive | — |

The code enforces steps 0 and 2: `ofs estimate` refuses real data before `prereg-v1`, and
the confirmation sample before `config-frozen` or if `config/estimation.yaml` changed since.
In the production image the tags are baked at build time
([ADR 0006](adr/0006-protocol-state-in-the-image.md)).

The full text is [config/preregistration.md](https://github.com/NIDHAL277/orderflow-scaling/blob/main/config/preregistration.md).

## Commit ids in the results

Every row of the estimates, tests and forecast tables records `code_version`, the commit
that computed it; the stamp ends in `+dirty` when tracked files had uncommitted changes,
and no result row has it. Before the first push to GitHub (18 November 2026) the history
was rewritten once, only to replace the author email with GitHub's private address: the
files, dates and messages of every commit are unchanged, but every commit id changed. The
results were computed earlier, so they record the earlier ids:

| `code_version` in the results | Public commit | Results |
|---|---|---|
| `e88fe171072d` | [`e01f7d96acd9`](https://github.com/NIDHAL277/orderflow-scaling/commit/e01f7d96acd9), tag `config-frozen` | confirmation estimates and tests, forecast confirmation tests, estimates history |
| `801063739473` | [`f14b95e57c3b`](https://github.com/NIDHAL277/orderflow-scaling/commit/f14b95e57c3b) | development estimates, forecast development tests |
| `b54080239763` | [`9e65d28dd17e`](https://github.com/NIDHAL277/orderflow-scaling/commit/9e65d28dd17e) | development tests |
| `2bc09c381596` | [`d40500f53f07`](https://github.com/NIDHAL277/orderflow-scaling/commit/d40500f53f07) | rolling estimates |

The confirmatory results were therefore computed at the commit tagged `config-frozen`
itself, with no uncommitted change.

## Timeline (12 weeks, ~15 h a week)

| Phase | Weeks | Gate |
|---|---|---|
| P0 Framing | S1 | CI green, protocol written |
| P1 Ingestion and reduction | S2–S3 | statistics = naive computation |
| P2 Simulation and estimators | S4–S5 | O2 met, tag `prereg-v1` |
| P3 Full universe, development | S6–S7 | tag `config-frozen` |
| P4 Confirmation and tests | S8–S9 | H1–H5 decided |
| P5 Production | S10 | daily pipeline online |
| P6 Report and dissemination | S11–S12 | report submitted, README ready |

More than two weeks late at the end of S7: cut the scope to 20 spot pairs and H1, H2 only.
