"""Physical layout of the data directory.

Every table is Parquet, partitioned by pair and month (spec §6.5). Per-day
outputs carry their stage version in the file name, ``{day}__v{version}.parquet``:
re-running a stage with the same version is a no-op, bumping the version
recomputes it (spec §7, idempotence). The daily statistics of closed months are
folded into one pack per month, ``{YYYY-MM}__pack.parquet`` (``ofs.packs``).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from ofs.config import Kind, Market, Sample


@dataclass(frozen=True)
class DataLayout:
    """Paths of every table and file below the data directory."""

    root: Path

    # ------------------------------------------------------------------ raw
    def raw_file(self, market: Market, kind: Kind, asset: str, day: dt.date) -> Path:
        """Return the downloaded daily file, kept until the day is reduced."""
        return self.root / "raw" / market / kind / asset / f"{asset}-{kind}-{day.isoformat()}.zip"

    def raw_month_file(
        self, market: Market, kind: Kind, asset: str, dataset_symbol: str, first_day: dt.date
    ) -> Path:
        """Monthly file of ``dataset_symbol``; a rename month has one file per symbol."""
        name = f"{dataset_symbol}-{kind}-{first_day:%Y-%m}.zip"
        return self.root / "raw" / market / f"{kind}_monthly" / asset / name

    @property
    def raw_dir(self) -> Path:
        """Folder of the downloaded files."""
        return self.root / "raw"

    # ------------------------------------------------------- bookkeeping
    @property
    def manifest_file(self) -> Path:
        """The download manifest."""
        return self.root / "manifest" / "manifest.parquet"

    def validation_file(self, asset: str, market: Market, day: dt.date) -> Path:
        """Return the validation report of one pair-day."""
        return (
            self.root
            / "validation"
            / f"symbol={asset}"
            / f"market={market}"
            / f"month={day:%Y-%m}"
            / f"{day.isoformat()}.parquet"
        )

    @property
    def validation_dir(self) -> Path:
        """Folder of the validation reports."""
        return self.root / "validation"

    @property
    def validation_cache(self) -> Path:
        """All validation reports in one file, refreshed from the per-day files on read."""
        return self.root / "cache" / "validation.parquet"

    # ------------------------------------------------------ partitioned
    def partition_dir(self, table: str, asset: str, market: Market, day: dt.date) -> Path:
        """Partition of ``table`` for a pair, market and month."""
        return self.root / table / f"symbol={asset}" / f"market={market}" / f"month={day:%Y-%m}"

    def day_file(self, table: str, asset: str, market: Market, day: dt.date, version: str) -> Path:
        """Per-day output of ``table`` at ``version``."""
        return (
            self.partition_dir(table, asset, market, day) / f"{day.isoformat()}__v{version}.parquet"
        )

    def day_files(self, table: str, asset: str, market: Market, day: dt.date) -> list[Path]:
        """All versions of a per-day output that currently exist."""
        folder = self.partition_dir(table, asset, market, day)
        if not folder.exists():
            return []
        return sorted(folder.glob(f"{day.isoformat()}__v*.parquet"))

    def pack_file(self, table: str, asset: str, market: Market, day: dt.date) -> Path:
        """Monthly pack of ``table`` for the month of ``day`` (``ofs.packs``)."""
        return self.partition_dir(table, asset, market, day) / f"{day:%Y-%m}__pack.parquet"

    def table_dir(self, table: str) -> Path:
        """Root folder of ``table``."""
        return self.root / table

    @staticmethod
    def bins_table(resolution_s: int) -> str:
        """Name of the table of windows of ``resolution_s`` seconds."""
        return f"bins_{resolution_s}s"

    @staticmethod
    def stats_table(kind: str) -> str:
        """Name of the table of daily statistics ``kind``."""
        return f"daily_stats/{kind}"

    # ------------------------------------------------------------ results
    def estimates_file(self, sample: Sample) -> Path:
        """Estimates of ``sample``."""
        return self.root / "estimates" / f"estimates_{sample}.parquet"

    def bootstrap_file(self, sample: Sample) -> Path:
        """Bootstrap replicates of the estimates of ``sample``."""
        return self.root / "estimates" / f"bootstrap_{sample}.parquet"

    def bootstrap_index_file(self, sample: Sample) -> Path:
        """Resampled days of each replicate of ``sample``."""
        return self.root / "estimates" / f"bootstrap_index_{sample}.parquet"

    def tests_file(self, sample: Sample) -> Path:
        """Cross-sectional tests of ``sample``."""
        return self.root / "tests" / f"tests_{sample}.parquet"

    def residuals_file(self, sample: Sample) -> Path:
        """Per-pair gaps to the predicted lines (forest plot)."""
        return self.root / "tests" / f"residuals_{sample}.parquet"

    def controls_file(self, sample: Sample) -> Path:
        """Negative controls and power of ``sample``."""
        return self.root / "tests" / f"controls_{sample}.parquet"

    def robustness_file(self, sample: Sample) -> Path:
        """Specification curve of ``sample``."""
        return self.root / "tests" / f"robustness_{sample}.parquet"

    # ------------------------------------------------------------ forecasting
    @property
    def forecast_dir(self) -> Path:
        """Forecasting extension outputs (``ofs forecast``)."""
        return self.root / "forecast"

    def forecasts_file(self, sample: str) -> Path:
        """Every out-of-sample forecast of ``sample`` with its realised target."""
        return self.forecast_dir / f"forecasts_{sample}.parquet"

    def forecast_eval_file(self, sample: str) -> Path:
        """Per-pair evaluation (mean losses, DM tests, model confidence set)."""
        return self.forecast_dir / f"evaluation_{sample}.parquet"

    def forecast_tests_file(self, sample: str) -> Path:
        """Pooled forecast hypotheses F1-F3 of ``sample``."""
        return self.forecast_dir / f"tests_{sample}.parquet"

    @property
    def simulation_dir(self) -> Path:
        """Simulation study outputs and checkpoints."""
        return self.root / "simulation"

    @property
    def quality_dir(self) -> Path:
        """Data-quality report (objective O1)."""
        return self.root / "quality"

    @property
    def logs_dir(self) -> Path:
        """Daily JSON event logs."""
        return self.root / "logs"

    @property
    def lock_file(self) -> Path:
        """Lock held by every command that writes the shared data (one writer at a time)."""
        return self.root / "status" / "pipeline.lock"

    @property
    def status_file(self) -> Path:
        """Status of the last daily run."""
        return self.root / "status" / "pipeline_status.json"

    @property
    def snapshot_file(self) -> Path:
        """Manifest of a results snapshot, present only in a directory unpacked from one."""
        return self.root / "snapshot.json"
