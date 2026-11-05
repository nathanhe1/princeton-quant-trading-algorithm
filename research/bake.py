"""Embed a coefficient JSON into a strategy file between the BEGIN/END COEF markers.
usage: python bake.py coef.json ../strategies/alpha.py"""
import json, re, sys
coef, path = sys.argv[1], sys.argv[2]
m = json.load(open(coef))
blob = json.dumps(m, separators=(",", ":"))
src = open(path).read()
new = re.sub(r"(# BEGIN COEF\n).*?(\n# END COEF)", lambda x: x.group(1) + 'COEF_JSON = r"""' + blob + '"""' + x.group(2), src, flags=re.S)
assert new != src or blob in src
open(path, "w").write(new); print("baked", coef, "(trained on", len(m["train"]), "sessions) into", path)
