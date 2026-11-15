# 2026-11-15: Forked pool workers hang on Linux

Cause: every process pool used the platform's default start method. On Linux that is fork, and a worker forked while a thread of polars, numba or the BLAS holds a lock waits for ever. Development ran on Windows, where workers are spawned, so nothing showed. It was found by running the CI job in a Linux container before the first push: the test suite stopped, using no CPU, on the only test that reduces a range of days with two jobs. The daily pipeline in its container and the weekly simulation job would have frozen the same way.

Fix: `ofs.utils.process_pool` builds every pool with the spawn context (estimates, forecasts, forecast simulation, reduction, simulation study). The CI job is now rehearsed in a Linux container before a push that changes how processes or files are handled.

Test added: tests/unit/test_misc.py::test_process_pools_spawn_their_workers
