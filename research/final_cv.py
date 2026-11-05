from cv import cv
if __name__ == "__main__":
    s, f = cv("../strategies/alpha.py")
    print(s.round(0).to_string(index=False))
    print("OOS per session: mk120 mean $%.0f (sd %.0f, min %.0f), mk240 mean $%.0f, positive %d/24" % (
        s.mk120.mean(), s.mk120.std(), s.mk120.min(), s.mk240.mean(), (s.mk120 > 0).sum()))
    for liq in ("MAKER", "TAKER"):
        x = f[f.liq == liq]; sh = x.q.abs().sum()
        print(f"{liq}: {sh/24:,.0f} sh/session, {x.mk120.sum()/sh:.2f} c/sh (120), {x.mk240.sum()/sh:.2f} c/sh (240)")
    print((f.groupby("sym").mk120.sum() / 2400).round(0).to_dict())
    print("self-trade penalty total $", f.pen.sum() / 100)
