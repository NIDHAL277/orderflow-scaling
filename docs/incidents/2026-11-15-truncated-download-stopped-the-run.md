# 2026-11-15: A truncated download stopped the venue comparison

Cause: one daily file of Bybit's archive arrived truncated four times in a row. A connection closed early ends `HTTPResponse.read(amt)` without an error, so only the gzip check noticed, with an `EOFError`. After the retries that error was raised again and reached the command line, where click takes it for the end of input at a prompt: the command printed "Aborted.", exited 1 without a traceback, and the comparison stopped at 239 of 300 days.

Fix: the size is compared with `Content-Length` and the gzip stream is read to its end; after five attempts the day counts as failed (`DownloadError`) and is skipped, the run goes on, and a later run tries the day again. The comparison reports the days used.

Test added: tests/unit/test_venues.py::test_a_file_that_cannot_be_fetched_skips_its_day
