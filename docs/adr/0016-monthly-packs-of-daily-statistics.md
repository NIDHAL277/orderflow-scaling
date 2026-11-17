# ADR 0016 — Monthly packs of the daily statistics

**Status**: accepted (2026-11-17)

**Context**: [ADR 0002](0002-versioned-daily-outputs.md) writes one file per statistic, pair
and day. Two years of 119 pair-markets and 11 statistics made about 955,000 files of a few
kilobytes each. On the laptop's hard disk the daily cycle of 2026-11-15 took about 11 hours,
nearly all of it spent opening those files: the rolling estimates read 8,030 files per pair,
and the weekly regime monitor read all 11 statistics when it needs one. Each file is also
one request for the weekly backup (Cloudflare R2's free tier counts a million uploads a
month) and one entry to copy when the data move to a server.

**Decision**: once a month can no longer change in normal operation (it ends more than
`max(catchup_days, recheck_days) + 5` days before the run's day: outside the daily replay and
the weekly recheck), its per-day files are folded into one pack per statistic, pair and
month, `{YYYY-MM}__pack.parquet`, whose Parquet metadata lists the days it holds and the
version of each (`ofs.packs`). The weekly run packs the months that just closed; `ofs compact`
packs everything. Writers keep writing per-day files, so ADR 0002 holds unchanged:

* a per-day file shadows the pack's rows for its day (it is newer: a manual backfill, a
  version bump); the next compaction folds it in;
* a day is done at a version when its per-day file at that version exists, or when it has
  no per-day file and the pack holds it at that version;
* invalidating a day (`ofs recheck`) deletes its per-day files and rewrites the pack without it;
* compaction writes the pack atomically, reads it back and compares it with the per-day
  files, and only then deletes them; an interruption leaves per-day files that shadow
  identical rows.

Only the 11 statistics the estimators read are packed. The windows (`bins_*`), the seasonal
sums and the 5-year history keep their layout. The regime monitor now loads only what the
principal H0 estimator reads (`H0_KINDS`: 3 of the 11 statistics).

**Consequences**: readers get exactly the frames they got from per-day files (one frame per
day, in date order). This was checked on real data before the conversion: for five
pair-markets, every statistic read from packs over the 23 closed months equalled the
per-day read, and the regime monitor gave the same blocks from 3 statistics as from 11 (and
the same as its last stored run). Opening a rolling window reads about 11 × (23 packs +
47 days) files instead of 8,030; the 955,000 files become about 30,000 packs plus the open
months' per-day files. A month packed by mistake while still open stays correct (per-day
files shadow it); the price is one rewrite of the pack at the next compaction.
