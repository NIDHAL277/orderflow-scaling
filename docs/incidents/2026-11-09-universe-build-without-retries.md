# 2026-11-09: Universe build without retries

Cause: rebuilding the universe (about 10 000 requests to the S3 listing and the monthly klines) failed on one dropped connection ("Server disconnected without sending a response"): unlike the downloader, the universe builder had no retry.

Fix: every request of the universe builder goes through the downloader's policy (5 attempts, exponential back-off, on transport errors and 408/425/429/5xx).

Test added: tests/unit/test_universe.py::test_build_universe_ranks_filters_and_maps_perps (a dropped connection then a 503 before the first listing)
