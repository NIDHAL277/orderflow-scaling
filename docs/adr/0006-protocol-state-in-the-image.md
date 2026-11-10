# ADR 0006 — Protocol state baked into the production image

**Status**: accepted (2026-11-08)

**Context**: `ofs.protocol` checks the git tags `prereg-v1` and `config-frozen` and that
`config/estimation.yaml` is unchanged since `config-frozen`. The production image has no
`.git` directory.

**Decision**: the release workflow passes, as build arguments, the protocol tags present
(`OFS_PROTOCOL_TAGS`) and the SHA-256 of `config/estimation.yaml` at `config-frozen`
(`OFS_FROZEN_CONFIG_SHA256`); the guard accepts them in place of git and compares the
hash with the configuration shipped in the image. `OFS_CODE_VERSION` carries the commit.

**Consequences**: the daily rolling estimates run in production only from images built
after `prereg-v1`; an image whose configuration drifted from the frozen one refuses
confirmatory runs.
