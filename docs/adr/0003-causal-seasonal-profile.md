# ADR 0003 — Causal hour-of-week profile

**Status**: accepted (2026-11-08)

**Context**: crypto has a strong weekly seasonality; every series is divided by its mean
profile per hour of the week (168 cells) before estimation (spec §5.1). A profile computed
on the whole sample would make daily updates rewrite all past statistics.

**Decision**: pass 1 stores per-day partial sums per cell (count, sum of V, sum of F²); the
profile used for day d is computed on the trailing 84 days ending at d (12 observations per
cell). Volume is divided by its mean profile, signed flow by its root-mean-square profile,
both normalised to mean one. For the first week of a series, the first 7 available days
are used (a documented look-ahead of at most 6 days).

**Consequences**: daily statistics never change once written; the profile adapts slowly to
regime changes. The specification curve includes a variant without deseasonalisation.
