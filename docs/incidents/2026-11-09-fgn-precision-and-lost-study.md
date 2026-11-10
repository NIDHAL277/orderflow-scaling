# 2026-11-09: fGn embedding failed at H = 0.9 and a study run was lost

Cause: the fGn autocovariance was computed as a plain second difference, which subtracts numbers of order 1e11 at lags of millions; the 365-day simulations at H = 0.9 got a negative circulant embedding and raised. `run_study` then stopped saving at the first failing cell while the process pool kept computing every queued cell, so 8 hours of the full study were lost.

Fix: cancellation-free autocovariance (`expm1`/`log1p`), exact white noise at H = 0.5; a failing cell is now logged (`cell_failed`) and the others are still checkpointed; the checkpoint key leaves out the settings decided from the study (roles, calibration factor, sample split).

Test added: tests/simulation/test_simulators.py::test_embedding_of_a_long_persistent_series_is_valid
