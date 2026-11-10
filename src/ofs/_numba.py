"""Typed wrapper around :func:`numba.njit` so that jitted functions keep their signature."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

import numba

_CACHE = os.environ.get("OFS_NUMBA_CACHE", "1") != "0"


def njit[F: Callable[..., Any]](func: F) -> F:
    """``numba.njit(cache=True, nogil=True)`` that type checkers see through."""
    return cast(F, numba.njit(cache=_CACHE, nogil=True)(func))
