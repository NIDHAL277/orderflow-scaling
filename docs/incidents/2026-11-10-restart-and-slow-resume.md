# 2026-11-10: A restart stopped the long jobs, and the resume was slow

Cause: the development machine was restarted from the Windows menu at 23:15, which ended the detached backfill and the two jobs waiting on it (the earlier lid close of 03:14 had done the same). On relaunch, the resume script first ran `ofs clean-temp`; walking about a million partition files took more than 30 minutes, while holding the data lock that the backfill needs, so nothing downloaded during that time.

Fix: the resume script relaunches the jobs immediately; leftover `*.tmp-*` and `*.part` files are harmless (no reader matches them: every table is read through `*.parquet`) and are cleaned later with `ofs clean-temp` when no job is writing. Finished days and simulation cells are skipped on resume, so the restart cost about an hour of downloads and nothing else. Operations: on the production VM the jobs run in containers with `restart: unless-stopped`; a laptop must not be restarted or closed during a backfill.

Test added: none (operational); the reader side is covered by tests/unit/test_misc.py::test_only_temporary_files_are_cleaned.
