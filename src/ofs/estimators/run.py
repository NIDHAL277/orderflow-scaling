"""Estimation of every exponent for every (pair, market) of a sample, with bootstrap CIs.

Outputs (spec Annex C): ``estimates_{sample}.parquet`` and the replicates
``bootstrap_{sample}.parquet`` used by the cross-sectional tests. Every
replicate of every pair uses the *same* resampled days (synchronized bootstrap).

Robustness variants computed here for the primary estimators: block length
(3 and 14 days), high / low volatility regimes (days above / below the median
cross-sectional RV), and without days carrying an extreme price jump.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import pickle
import shutil
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ofs.config import Market, Project, Sample, model_hash
from ofs.estimators.base import NAN_ESTIMATE, Estimate, Estimator, Features, FloatArray
from ofs.estimators.registry import HISTORY_TABLE, STATS_KINDS, build_registry, history_registry
from ofs.inference.bootstrap import confidence_interval, synchronized_weights
from ofs.io import read_parquets, write_parquet_atomic
from ofs.logs import stage_event
from ofs.paths import DataLayout
from ofs.schemas import BOOTSTRAP, ESTIMATES
from ofs.utils import as_float
from ofs.versioning import code_version

CONCORDANCE_MARGIN = 0.05


IntMatrix = np.ndarray[Any, np.dtype[np.int64]]


@dataclass(frozen=True)
class BootstrapPlan:
    """Resampled days shared by every pair and market; ``""`` is the main block length."""

    days: tuple[dt.date, ...]
    weights: dict[str, FloatArray]
    indices: dict[str, IntMatrix]

    @property
    def main(self) -> FloatArray:
        """Bootstrap weights of the main block length."""
        return self.weights[""]


def make_plan(
    project: Project, sample: Sample, reps: int | None = None, days: Sequence[dt.date] | None = None
) -> BootstrapPlan:
    """Draw the synchronized bootstrap plan of ``sample``, shared by every pair and market."""
    cfg = project.estimation
    day_list = tuple(days) if days is not None else tuple(cfg.split.period(sample).days())
    b = reps or cfg.inference.bootstrap.replications
    idx, W = synchronized_weights(
        len(day_list), cfg.inference.bootstrap.mean_block_days, b, cfg.seed, sample
    )
    weights: dict[str, FloatArray] = {"": W}
    indices: dict[str, IntMatrix] = {"": idx}
    for block in cfg.inference.bootstrap.sensitivity_days:
        label = f"block={block:g}"
        indices[label], weights[label] = synchronized_weights(
            len(day_list), block, b, cfg.seed, sample
        )
    return BootstrapPlan(day_list, weights, indices)


def replicates(
    est: Estimator,
    feats: Features,
    meta: dict[str, Any],
    plan: BootstrapPlan,
    label: str = "",
    mask: FloatArray | None = None,
) -> FloatArray | None:
    """Bootstrap replicates of ``est``: weighted sums (additive) or resampled series."""
    if est.bootstrap_mode == "series":
        if mask is not None:
            return None  # a spectral estimator needs contiguous days
        return np.array(
            [est.fit_resampled(meta, row).value for row in plan.indices[label]], dtype=np.float64
        )
    W = plan.weights[label] if mask is None else plan.weights[label] * mask
    return np.array([est.fit(s, meta).value for s in W @ feats.matrix], dtype=np.float64)


def load_bundle(
    layout: DataLayout,
    asset: str,
    market: Market,
    days: Sequence[dt.date],
    kinds: Sequence[str] = STATS_KINDS,
) -> dict[str, pl.DataFrame]:
    """Daily statistics of one (pair, market) restricted to ``days``."""
    lo, hi = min(days), max(days)
    wanted = {d.isoformat() for d in days}
    bundle: dict[str, pl.DataFrame] = {}
    for kind in kinds:
        table = layout.stats_table(kind)
        files: list[Path] = []
        month = dt.date(lo.year, lo.month, 1)
        while month <= hi:
            folder = layout.partition_dir(table, asset, market, month)
            if folder.exists():
                files += [f for f in folder.glob("*.parquet") if f.name[:10] in wanted]
            month = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        bundle[kind] = read_parquets(sorted(files))
    return bundle


def market_regimes(
    layout: DataLayout, assets: Sequence[str], market: Market, days: Sequence[dt.date]
) -> dict[dt.date, str]:
    """'high' / 'low' volatility days: cross-sectional median daily RV above / below its median."""
    frames = [load_bundle(layout, a, market, days, ("rv",))["rv"] for a in assets]
    rv = (
        pl.concat([f for f in frames if f.height], how="diagonal_relaxed")
        if any(f.height for f in frames)
        else pl.DataFrame()
    )
    if rv.height == 0:
        return {}
    daily = rv.group_by("date").agg(pl.col("rv_5m").median().alias("m")).sort("date")
    threshold = as_float(daily["m"].median(), 0.0)
    return {d: ("high" if m > threshold else "low") for d, m in daily.iter_rows()}


def day_masks(
    bundle: dict[str, pl.DataFrame], days: Sequence[dt.date], regimes: dict[dt.date, str]
) -> dict[str, FloatArray]:
    """Day subsets for robustness: volatility regimes, days without jumps."""
    masks: dict[str, FloatArray] = {}
    if regimes:
        masks["regime=high"] = np.array([1.0 if regimes.get(d) == "high" else 0.0 for d in days])
        masks["regime=low"] = np.array([1.0 if regimes.get(d) == "low" else 0.0 for d in days])
    summary = bundle.get("day_summary", pl.DataFrame())
    if summary.height:
        jump_days = set(summary.filter(pl.col("n_jump_seconds") > 0)["date"].to_list())
        masks["no_jump_days"] = np.array([0.0 if d in jump_days else 1.0 for d in days])
    return masks


def clean_json(obj: Any) -> Any:
    """Convert numpy scalars and non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return None if not math.isfinite(float(obj)) else float(obj)
    return obj


@dataclass
class AssetResult:
    """Estimate rows and bootstrap replicates of one pair and market."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    boot: list[tuple[int, str, str, str, str, str, float]] = field(default_factory=list)


def estimate_asset(
    project: Project,
    asset: str,
    market: Market,
    sample: Sample,
    plan: BootstrapPlan,
    regimes: dict[dt.date, str],
    *,
    bundle: dict[str, pl.DataFrame] | None = None,
    registry: list[Estimator] | None = None,
) -> AssetResult:
    """Every estimator of one pair and market, with bootstrap CIs and variants."""
    cfg = project.estimation
    layout = DataLayout(project.data_dir)
    if sample == "history":  # daily volatility only, from the long RV history (spec §4)
        from ofs.reduce.history import history_version, load_history_rv

        if bundle is None:
            rv = load_history_rv(layout, asset, market, plan.days, history_version(project))
            bundle = {HISTORY_TABLE: rv}
        registry = registry or history_registry(cfg)
    bundle = bundle if bundle is not None else load_bundle(layout, asset, market, plan.days)
    registry = registry or build_registry(cfg)
    masks = day_masks(bundle, plan.days, regimes)
    out = AssetResult()
    points: dict[tuple[str, str], Estimate] = {}
    ci_method = cfg.inference.bootstrap.ci_method
    sd_inflation = cfg.inference.bootstrap.sd_inflation

    def record(
        est: Estimator,
        variant: str,
        role: str,
        point: Estimate,
        reps: FloatArray | None,
        n_days: int,
    ) -> None:
        lo, hi = (
            confidence_interval(reps, point.value, method=ci_method, scale=sd_inflation)
            if reps is not None
            else (math.nan, math.nan)
        )
        diag = dict(point.diagnostics)
        if reps is not None:
            diag["bootstrap_finite"] = int(np.isfinite(reps).sum())
            diag["bootstrap_sd"] = float(np.nanstd(reps)) if np.isfinite(reps).any() else None
        out.rows.append(
            {
                "symbol": asset,
                "market": market,
                "sample": sample,
                "exponent": est.exponent,
                "estimator": est.name,
                "variant": variant,
                "role": role,
                "value": point.value if point.ok else None,
                "ci_low": lo,
                "ci_high": hi,
                "n_days": n_days,
                "diagnostics": diag,
            }
        )
        if reps is not None:
            out.boot.extend(
                (b, asset, market, est.exponent, est.name, variant, float(v))
                for b, v in enumerate(reps)
            )

    for est in registry:
        feats = est.features(bundle, plan.days)
        if feats.matrix.size == 0 or feats.n_days == 0:
            record(est, est.variant, est.role, NAN_ESTIMATE, None, 0)
            continue
        point = est.fit(feats.total(), feats.meta)
        if est.role in ("primary", "control"):
            points[(est.exponent, est.role)] = point
        reps = None
        if est.bootstrap and point.ok:
            reps = replicates(est, feats, est.warm_meta(feats.meta, point), plan)
        record(est, est.variant, est.role, point, reps, feats.n_days)
        if est.role != "primary" or not point.ok:
            continue
        meta_b = est.warm_meta(feats.meta, point)
        for label in plan.weights:
            if label:
                record(
                    est,
                    f"{est.variant}|{label}",
                    "robustness",
                    point,
                    replicates(est, feats, meta_b, plan, label),
                    feats.n_days,
                )
        if est.bootstrap_mode == "series":
            continue  # day-subset variants need contiguous days
        for label, mask in masks.items():
            if mask.sum() < 30:
                continue
            sub = est.fit(mask @ feats.matrix, feats.meta)
            sub_reps = (
                replicates(est, feats, est.warm_meta(feats.meta, sub), plan, mask=mask)
                if sub.ok
                else None
            )
            record(est, f"{est.variant}|{label}", "robustness", sub, sub_reps, int(mask.sum()))

    for row in out.rows:  # concordance of primary and control estimators (spec §5)
        if row["role"] != "primary" or row["variant"].count("|"):
            continue
        control = points.get((row["exponent"], "control"))
        if control is not None and row["value"] is not None and control.ok:
            diff = float(row["value"]) - control.value
            row["diagnostics"]["control_value"] = control.value
            row["diagnostics"]["concordant"] = abs(diff) <= CONCORDANCE_MARGIN
            if row["exponent"] in cfg.inference.concordance_exempt:  # deviation D2
                row["diagnostics"]["control_descriptive"] = True
    return out


_WORKER: dict[str, Any] = {}


def _init_worker(
    project: Project, plan: BootstrapPlan, regimes: dict[str, dict[dt.date, str]]
) -> None:
    _WORKER.update(project=project, plan=plan, regimes=regimes)


def _work(asset: str, market: Market, sample: Sample) -> AssetResult:
    return estimate_asset(
        _WORKER["project"],
        asset,
        market,
        sample,
        _WORKER["plan"],
        _WORKER["regimes"].get(market, {}),
    )


def results_frames(
    results: Sequence[AssetResult], project: Project
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the estimates and bootstrap tables, validated, with versions stamped."""
    now = dt.datetime.now(dt.UTC)
    version = code_version(project.root)
    cfg_version = model_hash(project.estimation)
    rows = [
        {
            **r,
            "diagnostics": json.dumps(clean_json(r["diagnostics"]), sort_keys=True),
            "code_version": version,
            "config_version": cfg_version,
            "computed_at": now,
        }
        for res in results
        for r in res.rows
    ]
    est = (
        ESTIMATES.validate(pl.DataFrame(rows, schema=ESTIMATES.columns))
        if rows
        else ESTIMATES.empty()
    )
    boot_rows = [b for res in results for b in res.boot]
    boot = (
        BOOTSTRAP.validate(pl.DataFrame(boot_rows, schema=list(BOOTSTRAP.columns), orient="row"))
        if boot_rows
        else BOOTSTRAP.empty()
    )
    return est, boot


def checkpoint_dir(
    layout: DataLayout, project: Project, sample: Sample, plan: BootstrapPlan
) -> Path:
    """Folder of the per-pair results of one estimation run.

    Its name hashes everything a result depends on (sample, days, replications,
    configuration, code version), so a checkpoint is only reused by the same run.
    """
    key = model_hash(
        [
            sample,
            [d.isoformat() for d in plan.days],
            int(plan.indices[""].shape[0]),
            model_hash(project.estimation),
            code_version(project.root),
        ],
        length=12,
    )
    return layout.root / "estimates" / "checkpoints" / f"{sample}-{key}"


class CheckpointStore:
    """Per-pair results saved as soon as they are computed, so a stopped run resumes.

    A laptop that sleeps or shuts down kills the estimation; before checkpoints every
    pair-market (about 3 hours of CPU in all) was lost, since results were written only
    at the end. Files are written atomically (temporary name, then rename).
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _path(self, asset: str, market: str) -> Path:
        return self.folder / f"{asset}__{market}.pkl"

    def load(self, asset: str, market: str) -> AssetResult | None:
        """Return the saved result of one pair-market, or None."""
        path = self._path(asset, market)
        if not path.exists():
            return None
        try:
            res = pickle.loads(path.read_bytes())
        except Exception:  # truncated or stale: recompute that pair
            path.unlink(missing_ok=True)
            return None
        return res if isinstance(res, AssetResult) else None

    def save(self, asset: str, market: str, res: AssetResult) -> None:
        """Save one pair-market's result atomically."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self._path(asset, market)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_bytes(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)

    def clear(self) -> None:
        """Remove the run's checkpoints once its files are written."""
        if self.folder.exists():
            shutil.rmtree(self.folder)


def run_estimates(
    project: Project,
    sample: Sample,
    markets: Sequence[Market] = ("spot", "um"),
    *,
    symbols: Sequence[str] | None = None,
    reps: int | None = None,
    jobs: int = 1,
    days: Sequence[dt.date] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate every exponent for every pair of the universe on one sample.

    Parameters
    ----------
    project : Project
        Configuration and data directory.
    sample : {'development', 'confirmation', 'rolling', 'history'}
        Sample whose days are used; ``history`` runs the daily H_sigma estimators on
        up to five years of daily realised variance.
    markets : sequence of {'spot', 'um'}
        Markets estimated.
    symbols : sequence of str, optional
        Restrict to these pairs.
    reps : int, optional
        Bootstrap replications (default: the configuration, 999).
    jobs : int
        Parallel processes.
    days : sequence of datetime.date, optional
        Days to use instead of the sample period (rolling estimates).

    Returns
    -------
    estimates, bootstrap : polars.DataFrame
        Also written to ``data/estimates/``.
    """
    layout = DataLayout(project.data_dir)
    plan = make_plan(project, sample, reps, days)
    universe = project.universe
    groups = [
        (p.symbol, m)
        for m in markets
        for p in universe.assets(m)
        if symbols is None or p.symbol in symbols
    ]
    regimes: dict[str, dict[dt.date, str]] = (
        {}
        if sample == "history"  # regimes come from the aggTrades RV of the 24-month window
        else {
            m: market_regimes(layout, [a for a, mm in groups if mm == m], m, plan.days)
            for m in markets
        }
    )
    store = CheckpointStore(checkpoint_dir(layout, project, sample, plan))
    done = {g: res for g in groups if (res := store.load(*g)) is not None}
    todo = [g for g in groups if g not in done]
    stage_event("estimate", "start", sample=sample, resumed=len(done), todo=len(todo))

    def finished(a: str, m: Market, res: AssetResult) -> None:
        store.save(a, m, res)
        done[(a, m)] = res
        stage_event(
            "estimate", "ok", symbol=a, market=m, sample=sample, done=len(done), of=len(groups)
        )

    if jobs > 1 and todo:
        with ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_worker, initargs=(project, plan, regimes)
        ) as pool:
            futures = {pool.submit(_work, a, m, sample): (a, m) for a, m in todo}
            for f in as_completed(futures):
                finished(*futures[f], f.result())
    else:
        for a, m in todo:
            finished(a, m, estimate_asset(project, a, m, sample, plan, regimes.get(m, {})))
    results = [done[g] for g in groups]  # spot by rank, then USD-M by rank
    est, boot = results_frames(results, project)
    write_parquet_atomic(est, layout.estimates_file(sample))
    write_parquet_atomic(boot, layout.bootstrap_file(sample))
    store.clear()  # the sample's files are written: the checkpoints have served
    main_idx = plan.indices[""]
    idx = pl.DataFrame(
        {
            "rep": np.repeat(np.arange(main_idx.shape[0]), main_idx.shape[1]),
            "position": np.tile(np.arange(main_idx.shape[1]), main_idx.shape[0]),
            "day": [plan.days[i] for i in main_idx.ravel()],
        }
    )
    write_parquet_atomic(idx, layout.bootstrap_index_file(sample))
    return est, boot
