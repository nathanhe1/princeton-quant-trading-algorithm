"""Shared loaders: wide per-batch frames of the top of book and trades."""
import json, os
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
SYMS = ["HPPO", "BNKY", "CRRT", "BRRY"]

def sessions():
    return json.load(open(os.path.join(DATA, "manifest.json")))["sessions"]

def tops(name):
    df = pd.read_csv(os.path.join(DATA, f"{name}.top.csv"))
    w = df.pivot(index="batch", columns="symbol", values=["bid_px", "ask_px", "bid_qty", "ask_qty"])
    out = {}
    for s in SYMS:
        b, a = w["bid_px"][s], w["ask_px"][s]
        bq, aq = w["bid_qty"][s], w["ask_qty"][s]
        out[s] = pd.DataFrame({"bid": b, "ask": a, "bq": bq, "aq": aq, "mid": (b + a) / 2,
                               "micro": (b * aq + a * bq) / (bq + aq), "spr": a - b})
    return out

def trades(name):
    return pd.read_csv(os.path.join(DATA, f"{name}.trades.csv"))
