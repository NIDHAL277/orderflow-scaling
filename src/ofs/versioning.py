"""Code and configuration versions stamped on every output.

* ``STAGE_VERSIONS`` are bumped by hand when an algorithm changes; the
  version of a per-day output is ``{stage}.{hash of the config it reads}`` so
  that editing a grid in ``config/`` recomputes exactly what depends on it.
* ``code_version`` is the git commit (``+dirty`` when the tree is modified),
  recorded next to every estimate (spec §10, reproducibility).
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ofs import __version__
from ofs.config import EstimationConfig, model_hash

STAGE_VERSIONS: dict[str, str] = {
    "validate": "3",  # 3: trade-id continuity on missing rows (ADR 0012)
    "bins": "1",
    "events": "1",
    "stats": "1",
    "klines_rv": "1",
    "history_rv": "3",  # 2: days without trades left out; 3: daily volume and trade count
    "flow_vc": "1",  # volume-centred flow variances (ADR 0015)
}


def stage_version(stage: str, *fragments: BaseModel | dict[str, Any] | list[Any]) -> str:
    """``{stage version}.{hash}`` of the configuration fragments a stage reads."""
    payload: list[Any] = [
        f.model_dump(mode="json", by_alias=True) if isinstance(f, BaseModel) else f
        for f in fragments
    ]
    return f"{STAGE_VERSIONS[stage]}.{model_hash(payload, length=8)}"


def events_version(cfg: EstimationConfig) -> str:
    """Version of pass-1 event-time statistics (sign ACF, runs, seasonal sums)."""
    return stage_version(
        "events",
        {
            "seed": cfg.seed,
            "sign_acf_max_lag": cfg.h0.sign_acf_max_lag,
            "sign_acf_lag_bins": cfg.h0.sign_acf_lag_bins,
        },
        cfg.impact.model_dump(
            mode="json", include={"proxy_a_gaps_s", "proxy_a_q_bins", "proxy_a_min_trades"}
        ),
    )


def stats_version(cfg: EstimationConfig) -> str:
    """Version of pass-2 statistics; it depends on pass 1 through the seasonal profile."""
    return stage_version(
        "stats",
        {
            "events": events_version(cfg),
            "windows": list(cfg.windows_seconds),
            "seasonality": cfg.seasonality,
            "seasonality_window_days": cfg.seasonality_window_days,
        },
        cfg.h0.model_dump(mode="json", include={"tau_grid", "dfa_boxes"}),
        cfg.hvol,
        cfg.hsigma,
        cfg.impact.model_dump(mode="json", include={"proxy_b_horizons_s", "proxy_b_x_edges"}),
    )


def flow_vc_version(cfg: EstimationConfig) -> str:
    """Version of the volume-centred flow variances (``flow_var_vc``): same inputs as pass 2."""
    return stage_version("flow_vc", {"stats": stats_version(cfg)})


@functools.cache
def code_version(root: Path | None = None) -> str:
    """Git commit of the working tree, or a fallback when git is unavailable."""
    env = os.environ.get("OFS_CODE_VERSION")
    if env:
        return env
    cwd = str(root) if root else None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return f"v{__version__}+nogit"
    return f"{head}+dirty" if dirty else head


def git_tag_exists(tag: str, root: Path | None = None) -> bool:
    """Whether ``tag`` exists in the git repository at ``root``."""
    try:
        out = subprocess.run(
            ["git", "tag", "--list", tag],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return out == tag


def config_unchanged_since(tag: str, path: str, root: Path | None = None) -> bool:
    """Tell whether ``path`` (e.g. ``config/``) is identical in the working tree and at ``tag``."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", tag, "--", path],
            cwd=str(root) if root else None,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0
