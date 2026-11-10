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

    def quality(self) -> pl.DataFrame:
        """Data-quality report per pair and market (``ofs quality``, objective O1)."""
        return self._read(self.layout.quality_dir / "quality.parquet")

    def simulation(self, profile: str) -> pl.DataFrame:
        """Summary of the simulation study ``profile``."""
        return self._read(self.layout.simulation_dir / f"summary_{profile}.parquet")

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
        """Pipeline health: status, last day, counts, failures, disk."""
        man = self.manifest()
        status = read_status(self.layout.status_file)
        counts: dict[str, int] = {}
        if man.height:
            counts = {str(k): int(v) for k, v in man.group_by("status").len().iter_rows()}
        last_day: dt.date | None = None
        if man.height and (man["status"] == "ok").any():
            last_day = man.filter(pl.col("status") == "ok")["date"].max()  # type: ignore[assignment]
        val = self.validation()
        failed = 0
        if val.height:
            failed = (
                val.filter(pl.col("blocking") & ~pl.col("passed"))
                .select(["symbol", "market", "date"])
                .unique()
                .height
            )
        return {
            "status": "ok" if not status.get("last_error") else "degraded",
            "last_ok_day": last_day.isoformat() if isinstance(last_day, dt.date) else None,
            "manifest_counts": counts,
            "validation_failures": failed,
            "disk_used_fraction": round(disk_fraction(self.layout.root), 4),
            "pipeline": status,
        }
