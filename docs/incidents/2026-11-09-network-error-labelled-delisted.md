# 2026-11-09: Network error labelled delisted

Cause: `explain_missing` relabelled network errors after the last available day as `delisted`, which would hide a failed download of the latest day from the O1 count.

Fix: only 404s (`missing`) are explained by the listing dates; errors stay unexplained until a retry succeeds.

Test added: tests/unit/test_download.py::test_manifest_explain_missing
