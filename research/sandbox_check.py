"""Held-out check on a recorded sandbox session (research/sandbox_data, from recorder.py): run a
strategy file with its built-in model on the recording, base and realistic stress, at given sizes.
Use once per candidate; do not tune on it.
    ../.venv/bin/python sandbox_check.py ../strategies/archive/alpha_v8c.py ../strategies/alpha.py
"""
import os
import sys

import evalx

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_data")
evalx.DATA = D

if __name__ == "__main__":
    import json
    names = json.load(open(os.path.join(D, "manifest.json")))["sessions"]
    for path in sys.argv[1:]:
        for scen in ("base", "share=0.3+through=2+adverse"):
            for size in (1.5, 0.5):
                for n in names:
                    s, f = evalx._one((os.path.abspath(path), None, n, scen, {"size_scale": size}, None, None))
                    mk = f[f.liq.str.contains("MAKER")]
                    tk = f[f.liq.str.contains("TAKER")]
                    c = lambda x: x.mk120.sum() / max(1, x.q.abs().sum())  # noqa: E731
                    print(f"{os.path.basename(path):14} {n} {scen[:22]:22} size {size}: ${s['mk120']:7.0f} "
                          f"| maker {mk.q.abs().sum():6.0f} sh {c(mk):+.2f}c | taker {tk.q.abs().sum():5.0f} sh {c(tk):+.2f}c",
                          flush=True)
