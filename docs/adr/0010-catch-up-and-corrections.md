# ADR 0010 — Daily catch-up window and application of re-published files

**Status**: accepted (2026-11-09)

**Context**: spec §11.4 asks to replay a day whose file was still absent at 12:00 UTC,
and §6.4 to apply the correction files published by Binance. The daily run processed
yesterday only, so a late file was never picked up. Binance Vision has no `updates`
folder: a corrected file replaces the old one at the same URL with a new `.CHECKSUM`, and
`ofs ingest --recheck` downloaded it again but left the statistics computed from the old
file in place (the day was already reduced, so it was skipped).

**Decision**:

* Each daily run processes a window of `catchup_days` (7) ending yesterday. Days already
  reduced cost one file-existence check; `missing` and `error` days are attempted again.
* `ofs recheck --from --to` (and the daily run after the `recheck_weekday`, over the last
  `recheck_days`) compares the published checksums with the manifest. For each changed
  pair-day it deletes the windows, statistics and validation report of that day, downloads
  both files again, validates and reduces the day, and recomputes pass 2 of the days whose
  seasonal profile includes it (the trailing 84 days after it, the first week before it).

**Consequences**: a late or corrected file reaches the statistics without manual work. A
full recheck of the 24-month window (about 170 000 checksum requests) is run by hand before
the configuration is frozen for the confirmation sample.
