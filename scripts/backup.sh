#!/usr/bin/env bash
# Weekly backup of the derived tables to Cloudflare R2 (S3-compatible) with rclone (spec §11.1).
# Raw Binance files are never backed up nor redistributed. rclone reads its remote from the
# environment (RCLONE_CONFIG_R2_*), see .env.example.
set -euo pipefail

DATA_DIR="${OFS_DATA_DIR:-/data}"
REMOTE="${RCLONE_REMOTE:-r2:orderflow-scaling}"

if [[ -z "${RCLONE_CONFIG_R2_ACCESS_KEY_ID:-}" ]]; then
  echo "backup skipped: R2 credentials are not configured" >&2
  exit 0
fi

# Every derived table: the statistics the daily run continues from, the results, the live
# forecasts and their track record (which cannot be recomputed), and what the dashboard
# shows besides (data quality, regime monitor, comparison of venues).
for table in daily_stats estimates tests forecast manifest validation simulation status \
    quality monitoring venues; do
  if [[ -d "${DATA_DIR}/${table}" ]]; then
    rclone sync "${DATA_DIR}/${table}" "${REMOTE}/${table}" --fast-list --transfers 8 --checksum
  fi
done
echo "backup done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
