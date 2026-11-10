import json, os
from concurrent.futures import ThreadPoolExecutor
from tgkrun import run
os.environ.setdefault("TGK_SCRATCH", "/private/tmp/claude-501/-Users-lulzx-work-tgk/6a0221b5-7095-4978-8293-d0dafa15348e/scratchpad")
ky = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
jobs = [(k, kap) for kap in [1.0, 1.4, 1.8] for k in ky]
def f(j):
    k, kap = j
    r = run(k, t_max=(300 if k <= 0.1 else 150), geometry="miller", kappa=kap, nz=32, nv=48, nmu=16, nthreads=1)
    return kap, {kk: float(v) for kk, v in r.items()}
out = {}
with ThreadPoolExecutor(5) as ex:
    for kap, r in ex.map(f, jobs):
        out.setdefault(str(kap), []).append(r); print(kap, r, flush=True)
json.dump(out, open("results/tgk_miller.json", "w"), indent=1)
