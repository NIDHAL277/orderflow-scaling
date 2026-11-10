# ADR 0001 — Record architecture decisions

**Status**: accepted (2026-11-08)

**Context**: the project is reviewed by quantitative researchers; choices that change a
number must be traceable.

**Decision**: every non-obvious technical or statistical choice gets a short record here:
context, decision, consequences. Statistical choices that affect the confirmatory tests are
also written in `config/preregistration.md` before the tag `prereg-v1`.

**Consequences**: a deviation after the tag needs both a new ADR and an entry in the
report's "Deviations from protocol".
