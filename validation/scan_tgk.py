import sys, json, os
from tgkrun import run
os.environ.setdefault("TGK_SCRATCH", "/private/tmp/claude-501/-Users-lulzx-work-tgk/6a0221b5-7095-4978-8293-d0dafa15348e/scratchpad")
name = sys.argv[1]
extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
ky = [round(0.05*i,3) for i in range(1,12)] if "ky" not in extra else extra.pop("ky")
out = {}
for res, kw in [("default", {}), ("refined", dict(nz=32, nv=48, nmu=16))]:
    rows = []
    for k in ky:
        o = dict(extra); o.update(kw); o.setdefault("nthreads", 4)
        r = run(k, t_max=(400 if k <= 0.1 else 200), **o)
        rows.append({kk: float(v) for kk, v in r.items()})
        print(res, rows[-1], flush=True)
    out[res] = rows
    json.dump(out, open(f"results/tgk_{name}.json", "w"), indent=1)
