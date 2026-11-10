# Operations

One virtual machine is enough: the pipeline runs once a day, updates the rolling
estimates and the dashboard. Target cost 0–5 € a month.

| Component | Choice | Note |
|---|---|---|
| Machine | Oracle Cloud "Always Free", Arm Ampere A1 | size for 2 cores / 12 GB; check the current free quota first |
| Backups | Cloudflare R2 via rclone | `scripts/backup.sh`, weekly |
| Monitoring | Healthchecks.io | ping each day; alert when missing |
| HTTPS | Caddy reverse proxy | automatic certificates, rate limit module |

Alternative: a laptop left on at night with the same `docker-compose.yml`.

## Deploy

```bash
git clone https://github.com/nidhal-v/orderflow-scaling && cd orderflow-scaling
cp .env.example .env            # Healthchecks URL, hostnames, R2 credentials
mkdir -p data && sudo chown 1000 data
docker compose pull && docker compose up -d
docker compose exec pipeline ofs status
```

## Daily run (`ofs daily`, cron `30 2-12 * * *`)

1. 02:30 UTC: download yesterday's files; if absent, exit code 75 and retry every hour.
   Each run also replays the six days before (late files, failed downloads).
2. Validation, reduction, update of the daily statistics. After Sunday's data, the files
   Binance re-published over the last 30 days are applied (`ofs recheck`).
3. Rolling 24-month estimates (confirmatory tests stay frozen on the frozen sample), then
   the live forecasts of every pair for the next 1, 5 and 22 days
   (`data/forecast/live/{day}.parquet`) and the live track record of past forecasts whose
   target is now complete (`data/forecast/live_scored.parquet`). Both need the protocol to
   allow the rolling estimates; a failure sets `live_forecasts: false` in the status and
   sends a fail ping, the data of the day being already saved. On the weekly recheck day,
   the change-point monitor of H0 runs too; pairs with an alarm in the latest block are
   listed in the status (`changepoint_alarms`) without failing the run.
4. Healthchecks.io ping on success, fail ping otherwise (and at 12:xx UTC for files still
   absent, which become "missing").
5. Sunday 04:00: backup of `daily_stats`, `estimates`, `tests` to R2.

Exit codes: 0 done (later runs of the same day are no-ops), 75 waiting for Binance or
another command writing the data, 1 failure.

One writer at a time: every command that writes the shared data (`ingest`, `backfill`,
`reduce`, `validate`, `recheck`, `history`, `estimate`, `test`, `daily`, `clean-temp`)
holds the operating-system lock `data/status/pipeline.lock`; a second writer exits at
once with code 75 (cron retries it the next hour). The lock belongs to the open file, so
a killed process never leaves it behind. `status`, `quality`, the API and the dashboard
only read and never wait.

## Backfill and data quality (O1)

```bash
ofs backfill --from 2024-11-01 --to 2026-10-31 --jobs 3   # resumable: done days are skipped
ofs quality  --from 2024-11-01 --to 2026-10-31            # exit 1 while O1 is not met
```

The backfill downloads chunk k+1 while chunk k is validated and reduced, and deletes each
raw file once reduced, so at most two chunks (a few GB) of raw files are on disk. It is
limited by bandwidth. Measured on 2026-11-09: 2024-11-01 for the 60 pairs and their
perpetuals was 0.3 GB (volatile days are larger: the BTCUSDT perpetual alone was 46 MB on
2025-03-10), the connection gave 3.1 MB/s with 1 or 6 parallel downloads, and the day was
reduced in 35 s with 4 processes. Reduced outputs take about 60 MB a day.

`ofs quality` writes `data/quality/day_states.parquet` (one row per pair, market and day)
and `data/quality/quality.parquet` (days per state and coverage per pair). Every day of the
window ends `reduced`, `blocked` (validation), or without a file: `not_listed`, `delisted`
or `suspended` are explained; `unexplained`, `not_reduced` and `not_attempted` are not, and
O1 requires none of them.

## Incidents

| Incident | Detection | Action |
|---|---|---|
| Daily file absent at 12:00 UTC | status `missing` in the manifest | replayed automatically by the next six daily runs; older gaps: `ofs backfill --from D --to D`; the dashboard shows the gap |
| Wrong checksum | M1 | new download; two failures alert |
| Binance format change | M2 schema failure | alert, fix the loader, add a regression test, replay |
| Disk full | alert at 80 % | purge `data/raw/`, check deletion after reduction |
| Interrupted run (kill, reboot) | `*.tmp-*` / `*.part` files | `ofs clean-temp --dry-run`, then `ofs clean-temp`; rerun the command: done days are skipped |
| Live forecasts failed | status `live_forecasts: false`, fail ping | read the `live_forecasts_failed` event in the log; the next daily run retries; a missing day of forecasts stays missing (forecasts are never made after the fact) |
| Regression after deployment | tests or dashboard | `OFS_IMAGE_TAG=<previous> docker compose up -d` |

Every real incident gets a three-line note in `docs/incidents/`: cause, fix, test added.
