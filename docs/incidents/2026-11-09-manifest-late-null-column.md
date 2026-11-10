# 2026-11-09: Manifest late null column

Cause: the first real one-day backfill crashed when writing the manifest: polars inferred the `error` column from the first 100 records (all null), then met "HTTP 404" (pairs not listed yet).

Fix: the manifest frame is built with the explicit schema of the contract; the other frames built from lists of rows read the whole list to infer types.

Test added: tests/unit/test_download.py::test_manifest_accepts_a_late_error
