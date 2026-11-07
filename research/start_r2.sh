#!/bin/bash
# Waits for R2 to be armed (or running), then starts the watchdog'd logger with R2's settings.
cd "$(dirname "$0")/.." && set -a && . ./.env && set +a
until .venv/bin/pqt check --server https://play.princeton-quant.com --key "$PQT_API_KEY_ALGO2" 2>&1 | grep -qE "R2 is (ARMED|RUNNING)"; do sleep 5; done
echo "$(date +%H:%M:%S) R2 visible, starting" >> research/sandbox_runs/watchdog_R2.log
PROBE_OVERRIDE='{"take_on": false, "news_trade": true}' TAG=R2 exec research/watchdog_live.sh
