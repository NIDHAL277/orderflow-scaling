# ADR 0004 — Additive daily statistics and the synchronized bootstrap

**Status**: accepted (2026-11-08)

**Context**: the bootstrap needs 999 re-estimations of every exponent for every pair,
with the same resampled days for all pairs, without re-reading trades.

**Decision**: every estimator consumes per-day sums (increments, squares, ACF sums,
moment sums, binned runs). Increments of a daily series (daily RV) are attached to their
start day, so they are additive too. A bootstrap replicate is a vector of day counts `w_b`
drawn once per sample by a stationary bootstrap (mean block 7 days) and applied to every
pair: `fit(w_b @ daily_matrix)`. The Whittle estimator, which needs a contiguous series,
re-estimates on the block-resampled series and drops increments across block joins.

**Consequences**: the bootstrap is a matrix product plus a fit; the cross-sectional
correlation of estimation errors between cryptos is preserved; replicates and day indices
are stored with the estimates for audit.
