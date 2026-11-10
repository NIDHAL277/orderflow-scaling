"""Healthchecks.io pings, disk alert and the pipeline status file (spec §11)."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any, Literal

import httpx

from ofs.io import read_json, write_json_atomic
from ofs.logs import get_logger

PingKind = Literal["start", "success", "fail"]
_SUFFIX: dict[str, str] = {"start": "/start", "success": "", "fail": "/fail"}


def ping(
    url: str | None,
    kind: PingKind,
    message: str = "",
    *,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """Ping Healthchecks.io; never raises (monitoring must not break the pipeline)."""
    if not url:
        return False
    try:
        if transport is not None:
            client = httpx.Client(transport=transport, timeout=10.0)
        else:
            from ofs.ingest.download import ssl_context

            client = httpx.Client(verify=ssl_context(), timeout=10.0)
        with client:
            resp = client.post(
                url.rstrip("/") + _SUFFIX[kind], content=message[:10_000].encode("utf-8")
            )
        return resp.status_code < 400
    except httpx.HTTPError as exc:
        get_logger().warning("healthcheck_ping_failed", kind=kind, error=repr(exc))
        return False


def disk_fraction(path: Path) -> float:
    """Return the used fraction of the disk holding ``path``."""
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return usage.used / usage.total


def write_status(path: Path, **fields: Any) -> dict[str, Any]:
    """Merge ``fields`` into the pipeline status file and return it."""
    status = read_json(path, default={}) or {}
    status.update(fields)
    status["updated_at"] = dt.datetime.now(dt.UTC).isoformat()
    write_json_atomic(status, path)
    return dict(status)


def read_status(path: Path) -> dict[str, Any]:
    """Read the pipeline status (empty when never written)."""
    data = read_json(path, default={})
    return dict(data) if isinstance(data, dict) else {}
