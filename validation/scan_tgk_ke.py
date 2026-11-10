"""Electrostatic kinetic-electron ITG (circular Miller, GX parameters), explicit RK4."""
import json, numpy as np, kerun
out = []
for ky in [0.1, 0.2, 0.3, 0.4, 0.5]:
    r = kerun.run(ky, beta=0.0, imex=0, nthreads=4); g, w = r["gamma"], r["omega"]
    print(f"ky={ky} gamma={g:.4f} omega={w:.4f}", flush=True)
    out.append(dict(ky=ky, gamma=g, omega=w))
json.dump(out, open("results/tgk_ke.json", "w"), indent=1)
