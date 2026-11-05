from cv import cv, summarize
if __name__ == "__main__":
    for te in [1, 3, 5, 8, 12]:
        s, f = cv("../strategies/alpha.py", {"make_on": 0, "take_edge": te})
        print(f"take_edge={te}: mk120/sess ${s.mk120.mean():.0f} mk240/sess ${s.mk240.mean():.0f}  sessions>0: {(s.mk120>0).sum()}/24")
        print(summarize(f)[["shares", "mk120", "c120", "c240"]].to_string())
