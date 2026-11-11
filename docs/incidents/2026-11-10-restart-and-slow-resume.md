# 2026-11-10: A restart stopped the long jobs, and the resume was slow

Cause: the development machine was restarted from the Windows menu at 23:15, which ended the detached backfill and the two jobs waiting on it (the earlier lid close of 03:14 had done the same). On relaunch, the resume script first ran `ofs clean-temp`; walking about a million partition files took more than 30 minutes, while holding the data lock that the backfill needs, so nothing downloaded during that time.

Fix: the resume script relaunches the jobs immediately; leftover `*.tmp-*` and `*.part` files are harmless (no reader matches them: every table is read through `*.parquet`) and are cleaned later with `ofs clean-temp` when no job is writing. Finished days and simulation cells are skipped on resume, so the restart cost about an hour of downloads and nothing else. Operations: on the production VM the jobs run in containers with `restart: unless-stopped`; a laptop must not be restarted or closed during a backfill.

Second occurrence: on 2026-11-11 at 04:42, after Windows had installed an update during the night, the machine was restarted again from the Start menu; the jobs stopped for eight hours until the next check. The relaunched backfill then spent about a minute per finished week just to skip it: each (pair, market) group read its 84-day seasonal profile before finding that every day was already reduced. `process_group` now reads the profile only when a day still needs pass 2, which turns the skip into a few file-existence checks. Operations: pause Windows Update while long jobs run.

Test added: none (operational); the reader side is covered by tests/unit/test_misc.py::test_only_temporary_files_are_cleaned.
