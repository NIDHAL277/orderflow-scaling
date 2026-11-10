"""Structured JSON logging: one event per (stage, pair, day) (spec §10).

Events go to stderr and, when a log directory is configured, are appended to
``data/logs/events-YYYY-MM-DD.jsonl`` where the dashboard reads them.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

import structlog

_LOG_DIR: Path | None = None
_LEVEL = "INFO"


class _JsonLineLogger:
    """Minimal structlog logger writing rendered JSON lines to stderr and a file."""

    def __init__(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def msg(self, message: str) -> None:
        sys.stderr.write(message + "\n")
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            path = self._log_dir / f"events-{dt.datetime.now(dt.UTC):%Y-%m-%d}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(message + "\n")

    debug = info = warning = warn = error = critical = exception = fatal = log = msg


def configure_logging(log_dir: Path | None = None, level: str = "INFO") -> None:
    """JSON events to stderr and, with ``log_dir``, to a daily file."""
    global _LOG_DIR, _LEVEL
    _LOG_DIR, _LEVEL = log_dir, level
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(sort_keys=True, default=str),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=lambda *_: _JsonLineLogger(_LOG_DIR),
        cache_logger_on_first_use=False,
    )


def logging_config() -> tuple[Path | None, str]:
    """Arguments of :func:`configure_logging` for worker processes (spawned, unconfigured)."""
    return _LOG_DIR, _LEVEL


def get_logger(**context: Any) -> Any:
    """Return a structlog logger bound to ``context``."""
    return structlog.get_logger().bind(**context)


def stage_event(
    stage: str,
    status: str,
    *,
    symbol: str | None = None,
    market: str | None = None,
    day: dt.date | None = None,
    **fields: Any,
) -> None:
    """Emit the canonical per-(stage, pair, day) event."""
    log = get_logger()
    method = log.error if status in {"error", "failed", "checksum_failed"} else log.info
    method(
        "stage",
        stage=stage,
        status=status,
        symbol=symbol,
        market=market,
        day=day.isoformat() if day else None,
        **fields,
    )
