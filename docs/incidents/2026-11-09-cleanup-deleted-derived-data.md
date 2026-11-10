# 2026-11-09: A cleanup command deleted derived data

Cause: before restarting the backfill, temporary files were removed with `Get-ChildItem -LiteralPath data -Recurse -Include '*.part','*.tmp-*' | Remove-Item`; Windows PowerShell 5.1 does not apply `-Include` together with `-LiteralPath`, so the command deleted every file it reached for about 13 minutes before it was stopped: in alphabetical order, all the 1-second and 10-second windows, the validation cache and most `day_summary` files; the ten other daily statistics tables, the validation reports, the manifest and the logs were not reached.

Fix: `ofs clean-temp` (with `--dry-run`) matches each file name in Python and deletes only `*.tmp-*` and `*.part`; operations use it instead of shell filters. The stages are idempotent: the restarted backfill finds pass 1 incomplete, downloads the raw files again from Binance Vision and rewrites the windows and `day_summary` (pass 2, intact, is kept); about eight hours of downloads were redone, nothing was lost.

Test added: tests/unit/test_misc.py::test_only_temporary_files_are_cleaned
