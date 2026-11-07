"""Unattended sandbox A/B campaign: one logger per sandbox session, each session running an
interleaved A/B (PROBE_ALTERNATE, 5-minute blocks). A new logger starts when a new session appears;
the old one is stopped (the sandbox sometimes restarts sessions early).
    ../.venv/bin/python campaign.py          (from research/; reads ../.env; runs until the list ends)
Each session's log: sandbox_runs/campaign_<session>.log and the probe's jsonl (its "start" event names
the session and the arms). Analyse with ab_report.py.
"""
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = dict(os.environ)
for line in open(ROOT / ".env"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        ENV[k] = v
PQT = str(ROOT / ".venv/bin/pqt")
CAP_OFF = {"_attrs": {"MAX_IMPROVE": None}}
CAP_ON = {"_attrs": {"MAX_IMPROVE": 2}}
PLAN = [  # one entry per session: the two arms (frozen v9: make_edge 2.5, cap 2, size 1.5, takes on)
    ("final", None),  # dress rehearsal: strategies/alpha.py as it is now, on its defaults, R1 command
    ("cap", [CAP_ON, CAP_OFF]),
    ("size", [{"size_scale": 1.5}, {"size_scale": 0.5}]),
    ("edge", [{"make_edge": 1.5}, {"make_edge": 2.5}]),
    ("takes", [{"take_on": True}, {"take_on": False}]),
    ("cap", [CAP_OFF, CAP_ON]),
    ("edge", [{"make_edge": 2.5}, {"make_edge": 1.5}]),
    ("size", [{"size_scale": 0.5}, {"size_scale": 1.5}]),
    ("edge", [{"make_edge": 1.5}, {"make_edge": 2.5}]),
]


def session():
    out = subprocess.run([PQT, "check", "--server", ENV["PQT_SERVER"], "--key", ENV["PQT_API_KEY_ALGO2"]],
                         capture_output=True, text=True, env=ENV).stdout
    m = re.search(r"session (\S+) is (\w+), batch (\S+)", out)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)


def main():
    seen = set(json.loads(os.environ.get("CAMPAIGN_SKIP", "[]")))
    proc = None
    i = 0
    while i < len(PLAN):
        name, state, batch = session()
        if name and name not in seen and state in ("ARMED", "RUNNING") and (state == "ARMED" or batch in ("None", "0") or int(batch) < 240):
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=30)
            subprocess.run(["pkill", "-INT", "-f", "sandbox_probe.py:Probe"])  # strays from earlier runs
            time.sleep(3)
            seen.add(name)
            tag, arms = PLAN[i]
            i += 1
            if arms is None:  # the current strategy file, no A/B
                env = {**ENV, "PROBE_BATCHES": "0", "PROBE_TAG": f"{tag}:{name}"}
            else:
                env = {**ENV, "PROBE_STRATEGY": str(ROOT / "research/campaign_alpha_v9.py"), "PROBE_BATCHES": "0",
                       "PROBE_ALTERNATE": json.dumps(arms), "PROBE_TAG": f"{tag}:{name}"}
            log = open(ROOT / f"research/sandbox_runs/campaign_{name}.log", "w")
            proc = subprocess.Popen([PQT, "run", "research/sandbox_probe.py:Probe", "--mode", "test",
                                     "--cancel-on-disconnect", "--server", ENV["PQT_SERVER"], "--key", ENV["PQT_API_KEY"]],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(time.strftime("%H:%M:%S"), "started", tag, "on", name, state, batch, flush=True)
        elif name in seen and proc and proc.poll() is not None:
            pass  # our logger ended with its session; wait for the next one
        time.sleep(20)
    if proc:
        proc.wait()
    print("campaign done", flush=True)


if __name__ == "__main__":
    main()
