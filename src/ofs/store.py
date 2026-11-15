"""Read-only access to the result tables for the API and the dashboard.

Parquet tables are loaded with polars and registered as DuckDB views in an
in-memory connection; queries take bound parameters. File paths are never
interpolated into SQL (the project path itself contains quotes).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from ofs.config import Sample
from ofs.ingest.manifest import explain_missing, read_manifest
from ofs.io import read_table_cached
from ofs.monitoring import disk_fraction, read_status
from ofs.paths import DataLayout

SAMPLES: tuple[Sample, ...] = ("development", "confirmation", "rolling", "history")


class Store:
    """Read-only access to the result tables (API and dashboard)."""

    def __init__(self, data_dir: Path) -> None:
        self.layout = DataLayout(data_dir)

    # ------------------------------------------------------------- loading
    def _read(self, path: Path) -> pl.DataFrame:
        return pl.read_parquet(path) if path.exists() else pl.DataFrame()

    def estimates(self, sample: Sample | None = None) -> pl.DataFrame:
        """Estimates of ``sample`` (all samples when None)."""
        samples = [sample] if sample else list(SAMPLES)
        frames = [
            f for f in (self._read(self.layout.estimates_file(s)) for s in samples) if f.height
        ]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def tests(self, sample: Sample | None = None) -> pl.DataFrame:
        """Tests of ``sample`` (all samples when None)."""
        samples = [sample] if sample else list(SAMPLES)
        frames = [f for f in (self._read(self.layout.tests_file(s)) for s in samples) if f.height]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def residuals(self, sample: Sample) -> pl.DataFrame:
        """Per-pair gaps of ``sample``."""
        return self._read(self.layout.residuals_file(sample))

    def robustness(self, sample: Sample) -> pl.DataFrame:
        """Specification curve of ``sample``."""
        return self._read(self.layout.robustness_file(sample))

    def controls(self, sample: Sample) -> pl.DataFrame:
        """Negative controls and power of ``sample``."""
        return self._read(self.layout.controls_file(sample))

    def manifest(self) -> pl.DataFrame:
        """Download manifest with 404s explained by the listing dates."""
        return explain_missing(read_manifest(self.layout.manifest_file))

    def validation(self) -> pl.DataFrame:
        """Every validation report (cached, see ``read_table_cached``)."""
        return read_table_cached(self.layout.validation_dir, self.layout.validation_cache)

    def validation_failures(self) -> int:
        """Pair-days that failed a blocking validation control (reads every report)."""
        val = self.validation()
        if not val.height:
            return 0
        return (
            val.filter(pl.col("blocking") & ~pl.col("passed"))
            .select(["symbol", "market", "date"])
            .unique()
            .height
        )

    def quality(self) -> pl.DataFrame:
        """Data-quality report per pair and market (``ofs quality``, objective O1)."""
        return self._read(self.layout.quality_dir / "quality.parquet")

    def simulation(self, profile: str) -> pl.DataFrame:
        """Summary of the simulation study ``profile``."""
        return self._read(self.layout.simulation_dir / f"summary_{profile}.parquet")

    # ------------------------------------------------------------- forecasting
    def forecasts(self, sample: str) -> pl.DataFrame:
        """Every out-of-sample forecast of ``sample`` (``ofs forecast``)."""
        return self._read(self.layout.forecasts_file(sample))

    def forecast_evaluation(self, sample: str) -> pl.DataFrame:
        """Per-pair forecast evaluation of ``sample``."""
        return self._read(self.layout.forecast_eval_file(sample))

    def forecast_tests(self, sample: str | None = None) -> pl.DataFrame:
        """Pooled forecast hypotheses F1-F3 (both samples when None)."""
        samples = [sample] if sample else ["development", "confirmation"]
        frames = [
            f for f in (self._read(self.layout.forecast_tests_file(s)) for s in samples) if f.height
        ]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def forecast_live(self) -> pl.DataFrame:
        """Live forecasts of the most recent origin (daily run)."""
        folder = self.layout.forecast_dir / "live"
        files = sorted(folder.glob("*.parquet")) if folder.exists() else []
        return self._read(files[-1]) if files else pl.DataFrame()

    def forecast_live_scored(self) -> pl.DataFrame:
        """Every live forecast with its realised target once complete (track record)."""
        return self._read(self.layout.forecast_dir / "live_scored.parquet")

    def changepoints(self) -> pl.DataFrame:
        """Block estimates of H0 with the change-point detectors (``ofs changepoints``)."""
        return self._read(self.layout.root / "monitoring" / "changepoints.parquet")

    def forecast_simulation(self) -> pl.DataFrame:
        """Forecast models on simulated rough volatility (``ofs simulate forecast``)."""
        return self._read(self.layout.simulation_dir / "forecast_study.parquet")

    def forecasts_for(
        self,
        symbol: str,
        sample: str,
        *,
        market: str = "spot",
        target: str = "volatility",
        horizon: int = 1,
        models: list[str] | None = None,
    ) -> pl.DataFrame:
        """Forecast and realised series of one pair (dashboard and API)."""
        fc = self.forecasts(sample)
        if fc.height == 0:
            return fc
        sql = (
            "select origin, model, forecast, realized, qlike, hurst from fc "
            "where symbol = $symbol and market = $market and target = $target "
            "and horizon = $horizon"
        )
        params: dict[str, Any] = {
            "symbol": symbol,
            "market": market,
            "target": target,
            "horizon": horizon,
        }
        out = self.query(sql + " order by origin, model", params, fc=fc)
        return out.filter(pl.col("model").is_in(models)) if models else out

    # ------------------------------------------------------------- queries
    def query(
        self, sql: str, params: dict[str, Any] | None = None, **tables: pl.DataFrame
    ) -> pl.DataFrame:
        """Run ``sql`` on the given frames registered as views (bound parameters only)."""
        con = duckdb.connect()
        try:
            for name, frame in tables.items():
                con.register(name, frame)
            out: pl.DataFrame = con.sql(sql, params=params).pl() if params else con.sql(sql).pl()
            return out
        finally:
            con.close()

    def estimates_for(
        self,
        symbol: str,
        sample: Sample | None = None,
        market: str | None = None,
        exponent: str | None = None,
    ) -> pl.DataFrame:
        """Estimates of one pair, optionally one sample, market and exponent."""
        est = self.estimates(sample)
        if est.height == 0:
            return est
        sql = "select * from est where symbol = $symbol"
        params: dict[str, Any] = {"symbol": symbol}
        if market:
            sql += " and market = $market"
            params["market"] = market
        if exponent:
            sql += " and exponent = $exponent"
            params["exponent"] = exponent
        sql += " order by sample, market, exponent, role, estimator, variant"
        return self.query(sql, params, est=est)

    def assets(self) -> pl.DataFrame:
        """Pairs present in the manifest with their last successfully downloaded day per market."""
        man = self.manifest()
        if man.height == 0:
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "market": pl.String,
                    "last_ok_day": pl.Date,
                    "days_ok": pl.Int64,
                }
            )
        return self.query(
            "select symbol, market, max(date) filter (where status = 'ok') as last_ok_day, "
            "count(*) filter (where status = 'ok') as days_ok, "
            "count(*) filter (where status in ('missing', 'error', 'checksum_failed')) as days_unexplained "
            "from man where kind = 'aggTrades' group by symbol, market order by symbol, market",
            man=man,
        )

    def health(self) -> dict[str, Any]:
        """Pipeline health: status, last day, counts, failures, disk.

        Cheap enough for a health check every minute: the status file and the manifest.
        The number of pair-days blocked by validation is the one the last daily run
        recorded; counting it reads every validation report (about 86,000 files for two
        years), see :meth:`validation_failures`.
        """
        man = self.manifest()
        status = read_status(self.layout.status_file)
        counts: dict[str, int] = {}
        if man.height:
            counts = {str(k): int(v) for k, v in man.group_by("status").len().iter_rows()}
        last_day: dt.date | None = None
        if man.height and (man["status"] == "ok").any():
            last_day = man.filter(pl.col("status") == "ok")["date"].max()  # type: ignore[assignment]
        recorded = status.get("validation_failures")
        failed = recorded if isinstance(recorded, int) else None
        return {
            "status": "ok" if not status.get("last_error") else "degraded",
            "last_ok_day": last_day.isoformat() if isinstance(last_day, dt.date) else None,
            "manifest_counts": counts,
            "validation_failures": failed,
            "disk_used_fraction": round(disk_fraction(self.layout.root), 4),
            "pipeline": status,
        }
