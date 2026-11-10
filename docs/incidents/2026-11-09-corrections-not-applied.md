# 2026-11-09: Re-published files not applied downstream

Cause: `ofs ingest --recheck` downloaded files whose `.CHECKSUM` had changed, but the day was already reduced, so its statistics kept coming from the old file; the daily run also never replayed a day that was missing at 12:00 UTC (found while auditing the pipeline against spec §6.4 and §11.4).

Fix: `ofs recheck` invalidates and recomputes every output of a corrected day and pass 2 of its neighbours; the daily run replays a 7-day window and applies re-published files weekly (ADR 0010).

Test added: tests/unit/test_reduce_pipeline.py::test_republished_file_is_applied_downstream, tests/unit/test_cli_daily.py::test_daily_replays_a_file_published_late
