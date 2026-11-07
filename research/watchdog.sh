#!/bin/bash
# Keeps the sandbox logger alive until the current session ends: if it exits early (dropped
# connection, SUPERSEDED, crash), restart it. Gives up after 3 restarts so two programs on one key
# don't keep knocking each other off. Every event goes to sandbox_runs/watchdog.log.
cd "$(dirname "$0")/.." && set -a && . ./.env && set +a
W=research/sandbox_runs/watchdog.log
sess() { .venv/bin/pqt check --server "$PQT_SERVER" --key "$PQT_API_KEY_ALGO2" 2>&1 | grep -o "session [^ ]* is [A-Z]*"; }
NAME=$(sess | awk '{print $2}'); restarts=0
echo "$(date +%H:%M:%S) watching $NAME" >> $W
while sess | grep -q "$NAME is RUNNING"; do
    if ! pgrep -f "sandbox_probe.py:Probe" >/dev/null; then
        restarts=$((restarts + 1))
        if [ $restarts -gt 3 ]; then echo "$(date +%H:%M:%S) GAVE UP after 3 restarts" >> $W; exit 1; fi
        LOG=research/sandbox_runs/rehearsal_$(date +%H%M%S).log
        echo "$(date +%H:%M:%S) logger not running: restart $restarts -> $LOG" >> $W
        PROBE_BATCHES=0 PROBE_TAG="rehearsal:$NAME" PROBE_OVERRIDE='{"make_edge": 1.5, "kill_floor": 2000}' \
            nohup .venv/bin/pqt run research/sandbox_probe.py:Probe --mode test --cancel-on-disconnect \
            --server "$PQT_SERVER" --key "$PQT_API_KEY" > "$LOG" 2>&1 &
    fi
    sleep 10
done
echo "$(date +%H:%M:%S) $NAME no longer running; watchdog done" >> $W
