#!/bin/bash
# R1 dress rehearsal on the sandbox: the RUNBOOK command (test mode), from the start of the next
# session, stopped when that session ends (one process trades one session).
cd "$(dirname "$0")/.." && set -a && . ./.env && set +a
LOG=research/sandbox_runs/rehearsal_$(date +%H%M).log
sess() { .venv/bin/pqt check --server "$PQT_SERVER" --key "$PQT_API_KEY_ALGO2" 2>&1 | grep -o "session [^ ]* is [A-Z]*, batch [^ ]*"; }
START=$(sess | awk '{print $2}')
[ "$NOW" = 1 ] && s=$(sess) || until s=$(sess) && [ "$(echo "$s" | awk '{print $2}')" != "$START" ] && echo "$s" | grep -qE "ARMED|batch (None|[0-9]{1,2})$"; do sleep 15; done
NAME=$(echo "$s" | awk '{print $2}'); echo "$(date +%H:%M:%S) starting on $s" | tee "$LOG"
PROBE_BATCHES=0 PROBE_TAG="rehearsal:$NAME" .venv/bin/pqt run research/sandbox_probe.py:Probe --mode test --cancel-on-disconnect \
    --server "$PQT_SERVER" --key "$PQT_API_KEY" >> "$LOG" 2>&1 &
PID=$!
until sess | grep -q "$NAME is ENDED" || ! sess | grep -q "$NAME"; do sleep 30; kill -0 $PID 2>/dev/null || break; done
sleep 20; kill -INT $PID 2>/dev/null; sleep 10; kill -KILL $PID 2>/dev/null
echo "$(date +%H:%M:%S) done" >> "$LOG"; echo "$LOG"
