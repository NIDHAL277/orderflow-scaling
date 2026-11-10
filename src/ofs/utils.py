"""Small typed helpers: polars scalars, process memory."""

from __future__ import annotations

import ctypes
import datetime as dt
import importlib
import math
import os
import sys


def as_float(value: object, default: float = math.nan) -> float:
    """Convert a polars/numpy scalar to ``float`` (``default`` for None or non-numbers)."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def as_date(value: object) -> dt.date | None:
    """Convert a date-like scalar to ``date`` (None otherwise)."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


class _MemoryCounters(ctypes.Structure):
    _fields_ = [  # PROCESS_MEMORY_COUNTERS (psapi.h)
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss_bytes() -> int:
    """Peak resident memory of this process, native allocations included (polars, numpy).

    ``tracemalloc`` only sees Python allocations; the spec's 8 GB limit is on the process.
    """
    if os.name == "nt":  # peak working set (psapi); explicit types for 64-bit handles
        win_dll = getattr(ctypes, "WinDLL")  # noqa: B009 - absent from non-Windows stubs
        kernel32, psapi = win_dll("kernel32"), win_dll("psapi")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MemoryCounters),
            ctypes.c_ulong,
        ]
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize)
    resource = importlib.import_module("resource")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) * (1 if sys.platform == "darwin" else 1024)  # bytes on macOS, KiB on Linux
