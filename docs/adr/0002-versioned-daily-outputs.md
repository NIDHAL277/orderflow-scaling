# ADR 0002 — Per-day outputs versioned in the file name

**Status**: accepted (2026-11-08)

**Context**: every stage must be idempotent and replayable for one pair and one day; a
change of algorithm or configuration must recompute exactly what depends on it.

**Decision**: per-day outputs are written as `{day}__v{version}.parquet` in
`symbol=/market=/month=` partitions, atomically (temporary file + rename). The version is
`{stage version}.{hash of the configuration fragment the stage reads}`; stage versions are
bumped by hand in `ofs.versioning.STAGE_VERSIONS`. A stage skips a day whose file with the
current version exists and replaces other versions of that day.

**Consequences**: re-running is free; editing a grid in `config/` recomputes pass 2 but not
pass 1 (which would need raw trades again). Results carry the git commit
(`code_version`) and the full configuration hash (`config_version`).
