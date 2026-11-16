# 2026-11-15: The health check read every validation report

Cause: `GET /health` counted the pair-days blocked by validation by reading every validation report, about 86,000 small files for two years. In the container, on a read-only mount of the data, the request had not answered after 400 s, while the health check of `docker-compose.yml` allows 10 s every minute: the API would have been marked unhealthy for ever. The unit tests ran on a handful of files and could not see it.

Fix: the daily run records the count in the status file (`Store.validation_failures` reads the reports once a day); `Store.health` returns the recorded number, or null before the first run, and reads only the status file and the manifest. The dashboard's health page read the same reports to list the failures and did not render in the container either: it now reads them on demand (a switch on the page), and a cached table is served even when the data are mounted read-only and the cache cannot be rewritten.

Test added: tests/unit/test_api_store.py::test_health_reports_the_count_the_daily_run_recorded, tests/unit/test_misc.py::test_cached_table_is_served_when_the_cache_cannot_be_written
