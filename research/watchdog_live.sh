#!/bin/bash
# R1/R2 on the competition server: the RUNBOOK command, restarted if it exits while the session is
# armed or running (max 3 restarts). Status checks use algo key 2 so they never displace key 1.
cd "$(dirname "$0")/.." && set -a && . ./.env && set +a
SRV=https://play.princeton-quant.com; TAG=${TAG:-R1}
W=research/sandbox_runs/watchdog_$TAG.log
sess() { .venv/bin/pqt check --server $SRV --key "$PQT_API_KEY_ALGO2" 2>&1 | grep -o "session [^ ]* is [A-Z]*"; }
restarts=-1
echo "$(date +%H:%M:%S) start: $(sess)" >> $W
while sess | grep -qE "$TAG is (ARMED|RUNNING|PAUSED)"; do
    if ! pgrep -f "sandbox_probe.py:Probe" >/dev/null; then
        restarts=$((restarts + 1))
        if [ $restarts -gt 3 ]; then echo "$(date +%H:%M:%S) GAVE UP after 3 restarts" >> $W; exit 1; fi
        LOG=research/sandbox_runs/${TAG}_$(date +%H%M%S).log
        echo "$(date +%H:%M:%S) launch $restarts -> $LOG" >> $W
        PROBE_BATCHES=0 PROBE_TAG="$TAG" nohup .venv/bin/pqt run research/sandbox_probe.py:Probe --mode live \
            --cancel-on-disconnect --server $SRV --key "$PQT_API_KEY" > "$LOG" 2>&1 &
    fi
    sleep 10
done
echo "$(date +%H:%M:%S) session over: $(sess)" >> $W
